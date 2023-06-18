from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open('C:\\Users\\Laksh-Games\\OneDrive\\Desktop\\Coding Files\\Py Stuff\\Supervised ML\\KNN\\knnmodel.pickle', 'rb'))

@app.route('/')
def home():
    return render_template("frontend.html")

@app.route("/recommend", methods=['GET','POST'])
def rec():
    if request.method == 'POST':
        pred = [[float(request.form.get('cg')),float(request.form.get('inventory')),float(request.form.get('netincome')),float(request.form.get('assetp')),float(request.form.get('assets')),float(request.form.get('debt')),float(request.form.get('ebit')),float(request.form.get('grossreven')),float(request.form.get('tl'))]]
        knnpredection = model.predict(pred)
        knnpredection = knnpredection[0]
        return render_template('frontend.html', predection='The company will still be '+knnpredection)
    return render_template('frontend.html', predection="Form Failed")

if __name__ == '__main__':
    app.run(debug=True)