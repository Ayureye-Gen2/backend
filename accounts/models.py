from django.db import models
from django.contrib.auth.models import AbstractUser, UserManager
from phonenumber_field.modelfields import PhoneNumberField


class UserProfile(AbstractUser):
    USER_CHOICES = [
        ("Gst", "Guest"),  # allowing just the preview
        ("Pt", "Patient"), # allowing fully featured inference and record viewing
        ("Dr", "Doctor"),  # allowing control over all the related patients' records
    ]
    user_type = models.CharField(
        max_length=3,
        choices=USER_CHOICES,
        default="Gst",
    )
    license_photo = models.ImageField(upload_to="licenses/", blank=True, null=True)
    is_verified = models.BooleanField(default=False) # if a doctor doesn't have a associated licence => not verified
    contact_number = PhoneNumberField(blank=True, null=True, unique=True) # Guest doens't need to have a contact number
    country = models.CharField(max_length=25, blank=True, null=True)
    city = models.CharField(max_length=25, blank=True, null=True)
    hospital_name = models.CharField(max_length=30, blank=True, null=True)



# class AyurEyeUserManager(UserManager):
#     def _create_user(self, username, email, password, **extra_fields):
#         if not email:
#             raise ValueError("Email must be set")
#         if not email:
#             raise ValueError("Username must be set")
#         email = self.normalize_email(email)
#         user = self.model(email=email, username=username, **extra_fields)
#         user.set_password(password)
#         user.save(using=self._db)
#         return user

#     def create_user(self, email,username, password=None, **extra_fields):
#         extra_fields.setdefault("is_staff", False)
#         extra_fields.setdefault("is_superuser", False)
#         return self._create_user(email,username, password, **extra_fields)

#     def create_superuser(self, email, username, password, **extra_fields):
#         extra_fields.setdefault("is_staff", True)
#         extra_fields.setdefault("is_superuser", True)

#         if extra_fields.get("is_staff") is not True:
#             raise ValueError("Superuser must have is_staff=True.")
#         if extra_fields.get("is_superuser") is not True:
#             raise ValueError("Superuser must have is_superuser=True.")

#         return self._create_user(email,username, password, **extra_fields)
