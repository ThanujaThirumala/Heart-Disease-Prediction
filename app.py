from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def predict():
    result = ''
    if request.method == 'POST':
        try:
            # Read all input values from form
            features = [float(x) for x in request.form.getlist('feature')]
            final_input = np.array([features])
            prediction = model.predict(final_input)
            result = 'Positive for Heart Disease' if prediction[0] == 1 else 'Negative for Heart Disease'
        except Exception as e:
            result = f"Error: {str(e)}"
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
