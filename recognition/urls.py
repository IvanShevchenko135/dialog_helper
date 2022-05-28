from django.urls import path

from . import views

urlpatterns = [
    path('api/transcript/', views.Transcript.as_view()),
    path('api/prediction/', views.Prediction.as_view()),
    path('api/start/', views.StartRecording.as_view()),
]
