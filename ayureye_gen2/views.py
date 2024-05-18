from django.shortcuts import render
from django.http import HttpResponse

from pathlib import Path
import re


def _convert_markdown_to_html(markdown: str):
    """ "
    Converts markdown formatted file to html formatted file

    For now this only works for headings

    TODO: Implement this properly
    """

    hashes_and_headings = {
        "#": 1,
        "##": 2,
        "###": 3,
        "####": 4,
        "#####": 5,
        "######": 6,
        "#######": 7,
    }

    def get_heading(match: re.Match) -> str:

        group: str = match.group(0)  # get all the match
        for key, value in reversed(
            hashes_and_headings.items()
        ):  # search from match containing more hashes
            if group.startswith(f"{key} "):  # there should be trailing space at the end
                headed_string = f"<h{value}> {group[value:]} </h{value}>\n"
                return headed_string

    return re.sub(
        "(" + ".+)|(".join([s for s in hashes_and_headings.keys()]) + ".+)",
        get_heading,
        markdown,
    ).replace("\n", "<br>")


def about_project(request):

    readme_path = Path(__file__).parents[1].joinpath("README.md")
    formatted_readme_content: str = """
    <html>

        <head> 
            <title> Failed Parsing </title>
        </head>

        <body>
            <h1> Error: Failed to parse the provided markdown. </h1>
        <body>

    </html>
  
    """

    with open(readme_path, "r") as readme:
        formatted_readme_content = _convert_markdown_to_html(readme.read())

    names_and_links = {"inference_api": "/inference"}

    # combine the names and references to get hyperlinks
    html_formatted_names_and_links = "\n".join(
        [
            "<a href =" + link + ">" + link_name + "</a>"
            for link_name, link in names_and_links.items()
        ]
    )

    html_response = f"""
        <html>
            <head>
                <title> About AyurGen-2 </title>
            </head>

            <body>
                {formatted_readme_content}
                <h2> Links </h2>
                {html_formatted_names_and_links}

            </body>
          
        </html>
    """

    return HttpResponse(html_response)
