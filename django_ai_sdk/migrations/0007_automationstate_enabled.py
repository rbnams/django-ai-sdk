from __future__ import annotations

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("django_ai_sdk", "0006_automations"),
    ]

    operations = [
        migrations.AddField(
            model_name="automationstate",
            name="enabled",
            field=models.BooleanField(blank=True, default=None, null=True),
        ),
    ]
