# Car-Value-Predictor
This project is a Car Price Predictor Web App built using Streamlit, designed to estimate the value of a car based on its specifications. It uses a trained Ridge Regression model along with a pre-fitted scaler to ensure accurate and consistent predictions.

The application provides an interactive and user-friendly interface where users can input vehicle details such as dimensions (wheelbase, car length, width, and curb weight), engine performance (engine size, horsepower, mileage), and configuration options (car body type, drive wheel, engine type, engine location, and number of cylinders). These inputs are aligned with the exact features used during model training, which helps prevent feature mismatch errors.

Categorical features are handled using manual one-hot encoding, ensuring that the input data structure matches the model’s expected format. Numerical features are scaled before prediction using a saved scaler, improving model performance.

Once the user clicks the prediction button, the app processes the data and displays the estimated car price in a visually appealing format. It also shows a metric summary for quick understanding. Additionally, the app includes a feature impact chart based on model coefficients, helping users understand which factors influence the predicted price the most.

The project follows good practices such as caching model loading for better performance, structured UI design using columns, and clear code organization for maintainability.

Overall, this project demonstrates how machine learning can be deployed in a practical and interactive way, providing real-time predictions with a clean and modern interface.
