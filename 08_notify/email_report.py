# financial_market_forcasting1/08_notify/email_report.py
"""
Mails you once a month, only when it matters: for the assets in settings.NOTIFY["assets"], only when the
call is in settings.NOTIFY["only_when"] (default: BUY) with at least the minimum confidence.
The mail explains itself: what to do, why (votes, the trusted model and its track record, what drives it),
what could happen to your money, how this rule did in the past, and how fresh the data is.

Every month's mail is saved as data/notify/last_email.html (also when nothing was sent), so you can see it
in the dashboard. Each asset is mailed at most once per month (data/notify/sent.csv); --fresh sends again.
Needs secrets.env (copy secrets.example.env). Without it, it only writes the preview.
"""
SETTINGS = {"enabled": True}

import smtplib
import ssl
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import pandas as pd

import settings
from core import store, verdict as V
from core.contracts import Model, Notifier
from core.plugins import load_players
from core.timeline import Timeline

COLOR = {"BUY": "#2e9e5b", "WAIT": "#d64545", "KEEP YOUR PLAN": "#c98a00"}


def secrets():
    p = settings.ROOT / "secrets.env"
    if not p.exists():
        return {}
    out = {}
    for line in p.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def asset_block(a, j, book, players, tl, amount):
    d = V.firsts(j, a)[-1]
    mt, situation = V.models_on(j, book, a, d)
    v = V.verdict(mt)
    lead = (v["who"] or list(mt.index[mt["trusted"]]) or [None])[0]
    why = [v["text"]]
    if lead:
        stats, _ = V.track_record(j, a, lead, d)
        if stats:
            why.append(f"{lead}'s track record on {a} since {stats['since']:%Y}: right direction {stats['right']:.0%} of the time "
                       f"(always 'up' would be {stats['always_up']:.0%}), result inside its range {stats['in_range']:.0%}.")
        try:
            why += [x for x in players[lead].explain(tl.view(d), a) if x.startswith("•")][:3]
        except Exception:
            pass
    mid, lo, hi = amount * (1 + v["p50"]), amount * (1 + v["p10"]), amount * (1 + v["p90"])
    ch = V.chance_up(v["p10"], v["p50"], v["p90"])
    past = V.history_summary(V.history(j, book, [a]))
    past_line = ""
    if v["call"] in past.index and "KEEP YOUR PLAN" in past.index:
        r, n = past.loc[v["call"]], past.loc["KEEP YOUR PLAN"]
        past_line = (f"In the past, when this rule said {v['call']} for {a} ({int(r['times'])} times), the next 20 trading days "
                     f"moved {r['avg_move_after']:+.1%} on average, vs {n['avg_move_after']:+.1%} in the other months.")
    action = {"BUY": f"Put money in {a} this month (at the next open).",
              "WAIT": f"Don't put money in {a} this month. Check again next month.",
              "KEEP YOUR PLAN": f"No forecast-based reason to change your plan for {a}."}[v["call"]]
    html = f"""
    <div style="border-left:6px solid {COLOR[v['call']]};padding:8px 14px;margin:14px 0;font-family:sans-serif">
      <h2 style="margin:4px 0;color:{COLOR[v['call']]}">{a.upper()}: {v['call']} <span style="font-size:15px;color:#555">confidence {v['confidence']}/100</span></h2>
      <p style="font-size:16px"><b>{action}</b></p>
      <p><b>Why</b></p><ul>{''.join(f'<li>{x.lstrip("• ")}</li>' for x in why)}</ul>
      <p><b>What could happen to EUR {amount:,.0f} in 20 trading days</b>: middle EUR {mid:,.0f}, bad case EUR {lo:,.0f},
         good case EUR {hi:,.0f}{f', chance of ending higher about {ch:.0%}' if pd.notna(ch) else ''}.</p>
      <p style="color:#555">{past_line} Situation: {situation}. Price EUR {mt['price'].iloc[0]:,.2f} on {d.date()}.</p>
    </div>"""
    return v, html


class EmailReport(Notifier):
    about = "monthly email, only when it matters (settings.NOTIFY)"

    def run(self, args):
        cfg = settings.NOTIFY
        j = store.read_csv(store.path("battle", "journal.csv"), dates=("date", "end_date"))
        book = store.read_csv(store.path("battle", "skillbook.csv"), dates=("as_of",))
        players = load_players(settings.ROOT / "04_models", Model, settings.HORIZON_DAYS)
        tl = Timeline.load()
        month = str(j["date"].max().to_period("M"))
        sent_p = store.path("notify", "sent.csv")
        sent = pd.read_csv(sent_p) if sent_p.exists() else pd.DataFrame(columns=["month", "asset", "call", "time"])
        blocks, calls, mail_assets = [], {}, []
        for a in cfg["assets"]:
            v, html = asset_block(a, j, book, players, tl, cfg["amount"])
            calls[a] = f"{v['call']} ({v['confidence']}/100)"
            blocks.append(html)
            already = ((sent["month"].astype(str) == month) & (sent["asset"] == a)).any()
            if v["call"] in cfg["only_when"] and v["confidence"] >= cfg["min_confidence"] and (not already or args.fresh):
                mail_assets.append(a)
        last = max(store.read_prices(a).index.max() for a in settings.ASSETS)
        body = (f"<div style='font-family:sans-serif;max-width:640px'><h1>financial_market_forcasting1, {pd.Period(month).strftime("%B %Y")}</h1>"
                + "".join(blocks) +
                f"<p style='color:#777;font-size:13px'>Prices up to {last.date()} (Yahoo Finance), forecasts made with only what was "
                f"known on the decision day. This is a test lab with virtual money, not financial advice. "
                f"Details: <a href='{cfg['dashboard_url']}'>{cfg['dashboard_url']}</a></p></div>")
        store.path("notify", "last_email.html").write_text(body)
        report = {"calls": calls, "preview": "data/notify/last_email.html"}
        if not mail_assets:
            report["mail"] = "not sent: nothing matches settings.NOTIFY this month"
            return report
        sec = secrets()
        if not sec.get("SMTP_USER"):
            report["mail"] = "not sent: no secrets.env yet (copy secrets.example.env)"
            return report
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"financial_market_forcasting1: {', '.join(f'{a.upper()} {calls[a]}' for a in mail_assets)}"
        msg["From"], msg["To"] = sec["SMTP_USER"], sec.get("MAIL_TO", sec["SMTP_USER"])
        msg.attach(MIMEText(body, "html"))
        with smtplib.SMTP_SSL(sec.get("SMTP_HOST", "smtp.gmail.com"), int(sec.get("SMTP_PORT", 465)),
                              context=ssl.create_default_context()) as s:
            s.login(sec["SMTP_USER"], sec["SMTP_PASSWORD"])
            s.send_message(msg)
        now = datetime.now().isoformat(timespec="seconds")
        sent = pd.concat([sent, pd.DataFrame([{"month": month, "asset": a, "call": calls[a], "time": now} for a in mail_assets])])
        sent.to_csv(sent_p, index=False)
        report["mail"] = f"sent to {msg['To']} about {', '.join(mail_assets)}"
        return report