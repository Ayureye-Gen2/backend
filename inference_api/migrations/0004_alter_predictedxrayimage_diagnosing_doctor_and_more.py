# Generated by Django 5.0.4 on 2024-06-04 08:58

import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('inference_api', '0003_alter_xrayprediction_original_image_and_more'),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.AlterField(
            model_name='predictedxrayimage',
            name='diagnosing_doctor',
            field=models.ForeignKey(default=7, limit_choices_to={'user_type': 'Dr'}, on_delete=django.db.models.deletion.DO_NOTHING, related_name='doctor_predicted_xray_image', to=settings.AUTH_USER_MODEL),
            preserve_default=False,
        ),
        migrations.AlterField(
            model_name='predictedxrayimage',
            name='patient',
            field=models.ForeignKey(default=5, limit_choices_to={'user_type': 'Pt'}, on_delete=django.db.models.deletion.CASCADE, related_name='patient_predicted_xray_image', to=settings.AUTH_USER_MODEL),
            preserve_default=False,
        ),
        migrations.AlterField(
            model_name='xrayimage',
            name='patient',
            field=models.ForeignKey(default=5, limit_choices_to={'user_type': 'Pt'}, on_delete=django.db.models.deletion.CASCADE, related_name='patient_xray_image', to=settings.AUTH_USER_MODEL),
            preserve_default=False,
        ),
    ]
