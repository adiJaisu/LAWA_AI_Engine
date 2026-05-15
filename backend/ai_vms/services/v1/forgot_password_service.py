"""
Defines the Forgot/Reset Password Service for handling users Passwords reset.

Includes:
- Generating Unique Verification Code and storing it for User.
- Sending Unique Verification Code to Users Email using Sendgrid.
- Verifying Sent Code inserted by User 
- Allowing User to reset the password and update it in the database

Author: HCLTech
"""
import os
import random
from typing import Union
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from fastapi import HTTPException
from ai_vms.models.users import User
from ai_vms.schemas.forgot_password import (
    SendVerificationCodeRequest,
    SendVerificationCodeSuccessResponse,
    SendVerificationCodeErrorResponse,
    VerifyVerificationCodeRequest,
    VerifyCodeSuccessResponse,
    VerifyCodeErrorResponse,
    ResetPasswordRequest,
    ResetPasswordSuccessResponse,
    ResetPasswordErrorResponse
    )
from ai_vms.utils.hashing_service import Hasher
from datetime import datetime, timedelta
from ai_vms.constant.constants import Constants as c
from ai_vms.config.config_manager import config
from ai_vms.config.logging_config import LoggingConfig
from ai_vms.utils.email_service import EmailService
from ai_vms.constant.email_templates import EmailTemplate as templates


logger = LoggingConfig().setup_logging()

class ForgotPasswordService:
    def __init__(self, db: Session):
        self.db = db

    def send_code(self, user_data: SendVerificationCodeRequest) -> Union[SendVerificationCodeSuccessResponse, SendVerificationCodeErrorResponse]:
        """
        Generates a verification code for the user (used for password reset), saves it in the database with an expiry time,
        and sends it via email.

        Args:
            user_data (SendVerificationCodeRequest): Request object containing the user's email.

        Returns:
            Union[SendVerificationCodeSuccessResponse, SendVerificationCodeErrorResponse]: Result of the operation.
        """
        try:
            logger.info("An API request to send verification code")

            # Validate email
            if not user_data.email:
                return SendVerificationCodeErrorResponse(code=500, message="Email Required!")

            # Fetch user by email
            user = self.db.query(User).filter(User.username == user_data.email).first()
            if not user:
                return SendVerificationCodeErrorResponse(code=404, message="User not found")

            # Generate code and set expiry
            code = self.generate_code()
            expiry_time = datetime.utcnow() + timedelta(minutes=10)  # OTP valid for 10 minutes

            # Update user record
            user.verify_code = code
            user.verify_code_expiry = expiry_time
            self.db.commit()

            # Send the code via email
            email_status = self.send_verification_email(user_data.email, code)
            if email_status:
                return SendVerificationCodeSuccessResponse()
            else:
                return SendVerificationCodeErrorResponse(code=500, message="Failed to send verification email.")

        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Database Error: {str(e)}")
            return SendVerificationCodeErrorResponse(code=500, message=f"Database Error: {str(e)}")

        except Exception as e:
            logger.error(f"Internal Server Error: {str(e)}")
            return SendVerificationCodeErrorResponse(code=500, message=f"Internal Server Error: {str(e)}")

    def generate_code(self):
        """
        Generates a 6-digit random verification code.

        Returns:
            int: A randomly generated integer between 100000 and 999999,
                typically used as a One-Time Password (OTP) for verification
        """
        return random.randint(100000, 999999)
    
    def send_verification_email(self, to_email: str, verification_code: str) -> bool:
        """
        Sends a verification email using SendGrid.
        
        Args:
            to_email (str): Recipient's email address.
            verification_code (str): The verification code to be sent.
        
        Returns:
            bool: True if the email was sent successfully (status code 202), False otherwise.
        """
        email_service = EmailService()

        subject = "AI-VMS App Verification Code"
        plain_text = f"Your verification code is: {verification_code}"
        html_content = templates.PASSWORD_RESET_TEMPLATE.format(
            verification_code=verification_code,
            year=datetime.now().year
        )

        return email_service.send_email(
            to_email=to_email,
            subject=subject,
            plain_text=plain_text,
            html_content=html_content
        )

    def verify_user_code(self, request: VerifyVerificationCodeRequest):
        """
        Verifies a user's submitted verification code against the stored one and checks if it has expired.

        Args:
            request (VerifyVerificationCodeRequest): The request object containing email and OTP code.

        Returns:
            VerifyCodeSuccessResponse | VerifyCodeErrorResponse
        """
        try:
            logger.info("An API request to verify user code")

            if not request.email or not request.code:
                return VerifyCodeErrorResponse(code=400, message="Email and verification code are required")

            user = self.db.query(User).filter(User.username == request.email).first()
            if not user:
                return VerifyCodeErrorResponse(code=404, message="User not found")

            if not user.verify_code or not user.verify_code_expiry:
                return VerifyCodeErrorResponse(code=400, message="No verification code found. Please request a new one.")

            if user.verify_code != request.code:
                return VerifyCodeErrorResponse(code=401, message="Invalid verification code")

            if datetime.utcnow() > user.verify_code_expiry:
                return VerifyCodeErrorResponse(code=403, message="Verification code has expired")

            # Setting the fields in database Null
            user.verify_code = None
            user.verify_code_expiry = None
            self.db.commit()

            return VerifyCodeSuccessResponse()

        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Database Error: {str(e)}")
            return VerifyCodeErrorResponse(code=500, message="Database Error")

        except Exception as e:
            logger.error(f"Internal Server Error: {str(e)}")
            return VerifyCodeErrorResponse(code=500, message="Internal Server Error")
        
    def reset_password(self, reset_data: ResetPasswordRequest) -> Union[ResetPasswordSuccessResponse, ResetPasswordErrorResponse]:
        """
        Resets the user's password by updating it with a newly hashed password.

        Args:
            reset_data (ResetPasswordRequest): Input data containing email and new password.

        Returns:
            ResetPasswordSuccessResponse: On successful password reset.
            ResetPasswordErrorResponse: On failure (e.g., user not found, password reused, database error).
        """
        try:
            logger.info(f"Attempting to reset password for {reset_data.email}")

            user = self.db.query(User).filter(User.username == reset_data.email).first()
            if not user:
                logger.error(f"User with email {reset_data.email} not found")
                return ResetPasswordErrorResponse(code=404, message="User not found")

            if Hasher.verify_password(reset_data.new_password, user.password):
                logger.warning(f"User attempted to reuse the old password for {reset_data.email}")
                return ResetPasswordErrorResponse(code=400, message="Please choose a different password")

            hashed_password = Hasher.get_hashed_password(reset_data.new_password)

            user.password = hashed_password
            user.last_updated_pass=datetime.utcnow()
            self.db.commit()

            logger.info(f"Password reset successfully for {reset_data.email}")
            return ResetPasswordSuccessResponse(code=200, message="Password has been reset successfully.")

        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Database error during password reset: {str(e)}")
            return ResetPasswordErrorResponse(code=500, message=f"Database Error: {str(e)}")

        except Exception as e:
            logger.error(f"Unexpected error during password reset: {str(e)}")
            return ResetPasswordErrorResponse(code=500, message=f"Internal Server Error: {str(e)}")