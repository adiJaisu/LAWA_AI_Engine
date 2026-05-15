"""
Defines application-wide email templates.

Author: HCLTech
"""
class EmailTemplate:

    PASSWORD_RESET_TEMPLATE = """
    <!DOCTYPE html>
    <html>
    <head>
        <style>
        body {{
            font-family: Arial, sans-serif;
            background-color: #f2f4f8;
            color: #333333;
            padding: 20px;
        }}
        .container {{
            background-color: #ffffff;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.05);
            max-width: 500px;
            margin: auto;
        }}
        .header {{
            text-align: center;
            font-size: 24px;
            font-weight: bold;
            color: #3e64ff;
            margin-bottom: 20px;
        }}
        .message {{
            font-size: 16px;
            line-height: 1.6;
        }}
        .code {{
            font-size: 28px;
            font-weight: bold;
            color: #3e64ff;
            text-align: center;
            margin: 30px 0;
            letter-spacing: 4px;
        }}
        .footer {{
            font-size: 12px;
            color: #888888;
            margin-top: 30px;
            text-align: center;
        }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">AI-VMS Support</div>
            <div class="message">
                Hello,<br><br>
                You recently requested to reset your password. Please use the verification code below to proceed:
            </div>
            <div class="code">{verification_code}</div>
            <div class="message">
                This code will expire shortly for your security.<br><br>
                If you did not request a password reset, you can safely ignore this message.
            </div>
            <div class="footer">
                &copy; {year} AI-VMS. All rights reserved.
            </div>
        </div>
    </body>
    </html>
    """
    
    ACCOUNT_ACTIVATED_TEMPLATE = """
    <!DOCTYPE html>
    <html>
    <head>
        <style>
        body {{
            font-family: Arial, sans-serif;
            background-color: #f2f4f8;
            color: #333333;
            padding: 20px;
        }}
        .container {{
            background-color: #ffffff;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.05);
            max-width: 500px;
            margin: auto;
        }}
        .header {{
            text-align: center;
            font-size: 24px;
            font-weight: bold;
            color: #3e64ff;
            margin-bottom: 20px;
        }}
        .message {{
            font-size: 16px;
            line-height: 1.6;
        }}
        .login {{
            font-size: 18px;
            font-weight: bold;
            text-align: center;
            color: #28a745;
            margin: 30px 0;
        }}
        .footer {{
            font-size: 12px;
            color: #888888;
            margin-top: 30px;
            text-align: center;
        }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">Welcome to AI-VMS 🎉</div>

            <div class="message">
                Hello {name},<br><br>

                Thank you for verifying your email address. We're excited to have you on board!  
                Your account has been successfully activated, and you're all set to begin your journey with us.
                <br><br>

            </div>

            <div class="login">
                You can now log in and start using AI-VMS App 🚀
            </div>

            <div class="message">
                If you have any questions or need assistance, feel free to reach out to our support team —
                we're here to help.
            </div>

            <div class="footer">
                &copy; {year} AI-VMS. All rights reserved.
            </div>
        </div>
    </body>
    </html>
    """