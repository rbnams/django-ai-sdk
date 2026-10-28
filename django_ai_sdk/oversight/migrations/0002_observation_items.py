from __future__ import annotations

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("django_ai_sdk_oversight", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="observation",
            name="items",
            field=models.JSONField(blank=True, default=dict),
        ),
    ]
