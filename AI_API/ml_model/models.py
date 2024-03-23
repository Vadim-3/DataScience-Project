from django.db import models
from .model.model import predict_image
import os


class Image(models.Model):
    class_choices = [
        (0, 'airplane'),
        (1, 'automobile'),
        (2, 'bird'),
        (3, 'cat'),
        (4, 'deer'),
        (5, 'dog'),
        (6, 'frog'),
        (7, 'horse'),
        (8, 'ship'),
        (9, 'truck'),
    ]
    image_class = models.IntegerField(choices=class_choices)
    image_predicted_class = models.IntegerField(choices=class_choices)
    image = models.ImageField(upload_to="img")

    def __str__(self):
        return f"{self.get_image_class_display()} - {self.id}"

    def save(self, *args, **kwargs):
        copied_image = self.image
        self.image_predicted_class = predict_image(copied_image)
        super().save(*args, **kwargs)
