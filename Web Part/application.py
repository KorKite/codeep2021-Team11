# Flask Module
from flask import Flask, url_for, flash ,redirect, render_template, request, abort, session
from flask_wtf import FlaskForm
from flask_wtf.file import FileField,FileAllowed,FileRequired
from werkzeug.utils import secure_filename
import os
from models import Model
import uuid

app = Flask(__name__)
app.secret_key = "23ijo#fkdlsj10#skldjf#!%"


class UploadForm(FlaskForm):
    input_img = FileField('image', validators=[FileRequired(), FileAllowed(['jpg','jpeg'], '.jpeg only!')])


@app.route("/", methods=('GET','POST'))
def index():
    form = UploadForm()
    if request.method=='POST' and form.validate_on_submit():
        filename = str(uuid.uuid4())[:15]
        input_url = os.path.join('static/images/input/', filename+ ".jpeg")
        input_img = form.input_img.data
        input_img.save(input_url)

        g_filename = filename+'-grid.jpeg'
        gradcam_url = os.path.join('./static/images/gradcam/', g_filename)
        information = Model().predict(input_url, gradcam_url)
        return render_template("predict.html", gradcam_url=gradcam_url, information=information)

    return render_template('index.html', form=form)




if __name__ == "__main__":
    app.run(debug=True, port=8891, host="0.0.0.0", threaded=True)