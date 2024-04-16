from django.shortcuts import render
from django.http import HttpResponse

from pathlib import Path

def about_project(request):
    """
    todo: apply markdown formatting
    """
    readme_path = Path(__file__).parents[1].joinpath("README.md")
    readme_content: str = ""
    with open(readme_path, "r") as readme:
        readme_content = readme.readlines()

    return HttpResponse(readme_content)
