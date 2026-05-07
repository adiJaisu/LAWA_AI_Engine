"""
Provides utility functions for password hashing and verification.

Includes:
- Secure password hashing using bcrypt.
- Password verification against stored hashes.

Author: HCLTech
"""

from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class Hasher:
    """
    Utility class for password hashing and verification using bcrypt.

    Author: HCLTech
    """

    @staticmethod
    def get_hashed_password(password: str) -> str:
        """
        Hashes a plain text password using bcrypt.

        Args:
            password (str): The plain text password to be hashed.

        Returns:
            str: The hashed password.
        """
        return pwd_context.hash(password)
    
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """
        Verifies a plain text password against a hashed password.

        Args:
            plain_password (str): The plain text password entered by the user.
            hashed_password (str): The stored hashed password.

        Returns:
            bool: True if the password matches, False otherwise.
        """
        return pwd_context.verify(plain_password, hashed_password)