#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import ssl
import smtplib
from pathlib import Path
from email.utils import formatdate
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.base import MIMEBase
from email import encoders

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
IMG = DOCS / "img"


def _must_env(name: str) -> str:
    v = os.environ.get(name, "").strip()
    if not v:
        raise SystemExit(f"Missing required env var: {name}")
    return v


def _attach_file(msg: MIMEMultipart, path: Path, mimetype=("application", "octet-stream")) -> None:
    if not path.exists():
        return
    part = MIMEBase(*mimetype)
    part.set_payload(path.read_bytes())
    encoders.encode_base64(part)
    part.add_header("Content-Disposition", f'attachment; filename="{path.name}"')
    msg.attach(part)


def main() -> None:
    mail_to = _must_env("MAIL_TO")
    mail_username = _must_env("MAIL_USERNAME")
    mail_password = _must_env("MAIL_PASSWORD")
    subject = os.environ.get("MAIL_SUBJECT", "Daily Ticker Report")

    to_list = [x.strip() for x in mail_to.split(",") if x.strip()]

    html_path = DOCS / "email.html"
    txt_path = DOCS / "email.txt"

    if not html_path.exists():
        raise SystemExit("docs/email.html missing (email HTML body not generated)")
    html = html_path.read_text(encoding="utf-8", errors="ignore")

    txt = ""
    if txt_path.exists():
        txt = txt_path.read_text(encoding="utf-8", errors="ignore")

    # Inline macro charts via CID (works even if remote URLs fail)
    cid_map = {
        "macro_vix_5y": IMG / "macro_vix_5y.png",
        "macro_eurusd_5y": IMG / "macro_eurusd_5y.png",
    }

    for cid, p in cid_map.items():
        # Replace any src="...macro_vix_5y.png" with src="cid:macro_vix_5y"
        html = re.sub(
            rf'src="[^"]*{re.escape(p.name)}"',
            f'src="cid:{cid}"',
            html,
            flags=re.IGNORECASE,
        )

    msg = MIMEMultipart("related")
    msg["From"] = mail_username
    msg["To"] = ", ".join(to_list)
    msg["Date"] = formatdate(localtime=True)
    msg["Subject"] = subject

    alt = MIMEMultipart("alternative")
    msg.attach(alt)

    if txt:
        alt.attach(MIMEText(txt, "plain", "utf-8"))
    alt.attach(MIMEText(html, "html", "utf-8"))

    # Attach inline images
    for cid, p in cid_map.items():
        if not p.exists():
            continue
        img = MIMEImage(p.read_bytes())
        img.add_header("Content-ID", f"<{cid}>")
        img.add_header("Content-Disposition", "inline", filename=p.name)
        msg.attach(img)

    # Attach report + state
    _attach_file(msg, DOCS / "report.md", ("text", "markdown"))
    _attach_file(msg, DOCS / "state.json", ("application", "json"))

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(mail_username, mail_password)
        server.sendmail(mail_username, to_list, msg.as_string())

    print("Email sent with inline macro charts.")


if __name__ == "__main__":
    main()
