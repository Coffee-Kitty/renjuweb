from flask import Blueprint, request, jsonify, redirect, send_file


front_api = Blueprint('front_api', __name__)

@front_api.route("/")
def default():
    return redirect("./auth.html")


@front_api.route("/<path:filename>.html")
def html_files(filename):
    directory = "./templates/"
    return send_file(f"{directory}{filename}.html")


@front_api.route("/<path:filename>.css")
def css_files(filename):
    directory = "./templates/"
    return send_file(f"{directory}{filename}.css")


@front_api.route("/<path:filename>.js")
def js_files(filename):
    directory = "./templates/"
    return send_file(f"{directory}{filename}.js")

@front_api.route("/<path:filename>.jpg")
def image_files(filename):
    directory = "./templates/"
    return send_file(f"{directory}{filename}.jpg")
