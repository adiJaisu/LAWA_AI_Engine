"""
Provides utility functions for sending email for various in-app operations
using Client's SMTP relay servers.

Author: HCLTech
"""

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime
from ai_vms.config.logging_config import LoggingConfig
from ai_vms.config.config_manager import config
import ai_vms.constant.email_templates as templates  # Your email templates

logger = LoggingConfig().setup_logging()


class EmailService:
    def __init__(self):
        # Client SMTP relay servers (IP based, no authentication)
        self.smtp_servers = [
            (config.SMTP_SERVER1_URL, config.SMTP_SERVER_PORT),
            (config.SMTP_SERVER2_URL, config.SMTP_SERVER_PORT),
        ]
        self.sender_email = config.SENDER_EMAIL  # still using as "from" email

    def send_email(self, to_email: str, subject: str, plain_text: str, html_content: str):
        """
        Sends an email using Client's SMTP relay servers (no authentication).

        Args:
            to_email (str): Recipient's email address.
            subject (str): Subject of the email.
            plain_text (str): Plain text content.
            html_content (str): HTML content.

        Returns:
            bool: True if the email was sent successfully, False otherwise.
        """
        # Create email message
        message = MIMEMultipart("alternative")
        message["From"] = self.sender_email
        message["To"] = to_email
        message["Subject"] = subject

        # Attach plain text and HTML versions
        if plain_text:
            message.attach(MIMEText(plain_text, "plain"))
        if html_content:
            message.attach(MIMEText(html_content, "html"))

        # Try both SMTP servers
        for host, port in self.smtp_servers:
            try:
                with smtplib.SMTP(host, port, timeout=10) as server:
                    # No login since it's IP-based authentication
                    server.sendmail(self.sender_email, to_email, message.as_string())

                logger.info(f"Email sent successfully to {to_email} via {host}:{port}")
                return True

            except Exception as e:
                logger.error(f"Error sending email via {host}:{port} to {to_email}: {e}")

        return False
