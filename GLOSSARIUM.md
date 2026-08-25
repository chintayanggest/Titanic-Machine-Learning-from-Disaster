# 🧠 The Glossarium
*A living repository of Data Science strategies, mental models, and engineering rules collected during the Titanic Machine Learning Project.*

---

## 🧭 Phase 1: Data Philosophy & Preparation
* **Domain Knowledge:** Background knowledge of the specific field the data belongs to (e.g., knowing the history of the Titanic). It helps Data Scientists make smart assumptions before looking at the data.
* **Hypothesis Generation:** The process of making educated guesses about your data before analyzing it. (e.g., "Women and children likely survived more"). You must make assumptions *before* writing a single line of Python, and then use EDA to prove them right or wrong.
* **"Killing Your Darlings":** No matter how clever or logical a hypothesis seems in your head, if the data does not show a strong, undeniable signal, you must be willing to throw the idea away. Keeping useless features adds "noise" to the model.
* **The "Honest" Baseline:** In Kaggle, many top scores rely on "data leaks" (like matching last names across train/test sets) or outright cheating. An honest, robust ML pipeline on raw demographic data usually scores lower than highly engineered "hacks"—but the honest pipeline is what companies actually want to see.
* **The Accuracy Ceiling (Aleatoric Uncertainty):** In real-world data, 100% accuracy is impossible because of randomness/luck. (e.g., A 3rd-class passenger surviving by pure chance). A true, honest Titanic model maxes out around 83-84%.

---

## 📊 Phase 2: Exploratory Data Analysis (EDA)
* **Outlier Detection (Boxplots):** Boxplots are the best tool to see the "spread" of your data. The box shows where the middle 50% of your data lives. Dots floating outside the "whiskers" are mathematical outliers that might confuse your model.
* **The Limits of Correlation Heatmaps:** Heatmaps (specifically Pearson correlation) measure the linear relationship between *numbers*. Text/Categorical data will be completely ignored unless you "encode" it (turn text into numbers) first.
* **Positive vs. Negative Correlation:** A positive correlation (e.g., 0.54) means as Feature A goes up, Feature B goes up. A negative correlation (e.g., -0.34) means as Feature A goes up, Feature B goes down. Both are equally powerful for a model.
* **Multicollinearity:** When two features in your dataset are highly correlated with *each other* (not the target). It means they are sharing redundant information. It's something Senior Data Scientists always keep an eye on.

---

## 🧹 Phase 3: Handling Missing Data
* **Missingness is Information:** Data isn't always missing by accident. Sometimes the reason it is missing is a powerful predictor (e.g., a person died and their records were lost). Instead of erasing the fact that it was missing, you can create a flag (like `Age_Is_Missing = 1`) as a clue.
* **The Hierarchy of Missing Data:**
    1. *Few rows missing (e.g., 2 in Embarked):* Fill with the most frequent value (mode).
    2. *Moderate amount missing (e.g., 177 in Age):* Use strategic imputation.
    3. *Vast majority missing (e.g., 687 in Cabin):* Drop the column entirely.
* **Strategic Guessing (Smart Imputation):** Never fill missing information with a blind global guess (mean/median). Look for hidden "proxy" variables (e.g., extracting "Titles" to group passengers, then filling a missing "Age" with the median age of that specific title group).
* **The Golden Rule of Imputation:** Always visualize the distribution of a feature *before* you fill missing values, and *after* you fill them. If the shape of your chart changes drastically, your imputation method was bad and introduced bias.

---

## ⚙️ Phase 4: Feature Engineering
* **Feature Mapping / Replacing:** The act of standardizing messy text data (e.g., teaching the computer that "Mlle" and "Ms" mathematically mean the exact same thing as "Miss").
* **Feature Consolidation:** If two columns represent the exact same concept (like `SibSp` and `Parch` both representing "Family Members"), models perform better when you combine them into a single, stronger feature (like `FamilySize`).
* **The Problem of Magnitude:** ML models are easily tricked by big numbers. If a model sees `Fare = 100` and `Age = 2`, it might think Fare is 50x more important. Fix this by Scaling or Binning.
* **The Scaling vs. Binning Tradeoff:**
    * *Binning (Discretization):* Great for things with natural "stages" (like Age: Child -> Teen -> Adult). It helps models ignore minor noise and neutralizes massive outliers.
    * *Scaling:* Better for precise, continuous measurements (like Fare, Weight) where the exact gaps between numbers hold valuable information.
* **StandardScaler vs. MinMaxScaler:** `MinMaxScaler` shrinks data strictly between 0 and 1. `StandardScaler` (Z-score) puts the mean average at 0, meaning below-average values become negative. `StandardScaler` handles wild outliers a bit better.
* **Protecting the Model from Over-Memorizing:** If you leave rare, unique data (like a one-of-a-kind title "Jonkheer") alone, the model might memorize that specific person's fate. Grouping rare things together helps the model learn broad *patterns*, not individual names.

---

## 🤖 Phase 5: Machine Learning & Tuning
* **Data Leakage (The Deadliest Sin):** Never calculate means, medians, or scalers on the *combined* train+test data. The model must learn *only* from the training data, treating the test data as a total mystery.
* **Cross-Validation (CV):** Never trust a model tested on the exact same data it learned from. CV internally slices your training data into smaller "train/test" splits, proving the model can handle unseen data.
* **The "Seed" (`random_state`):** Algorithms use randomness. To force the computer to be consistent so you can compare models fairly, set a `random_state` (usually 42).
* **Parameters vs. Hyperparameters:** *Parameters* are rules the model learns on its own (like "Women survive"). *Hyperparameters* are the settings you control before training starts (like "How deep should the decision tree grow?").
* **GridSearchCV (Automated Tuning):** A brute-force search technique that tests every single combination of hyperparameters to find the mathematical optimum. 
* **The Brute-Force Trap:** `GridSearchCV` is perfect for small datasets (like Titanic). On big data, it will crash your computer. For massive datasets, use `RandomizedSearchCV` or `Optuna`.
* **Overfitting to the Validation Set:** If you tune hyperparameters too aggressively, your model will eventually memorize the practice exams and fail on real-world data. Sometimes a simpler model is better.

---

## 🚀 Phase 6: Production & Sanity Checks
* **`fit_transform` vs `transform`:** 
    * `fit` = "learn the rules"
    * `transform` = "apply the rules"
    * You ONLY ever `fit` on the Training data. For Test data, you ONLY `transform`.
* **Object Memory (The Scaler Rule):** Writing `scaler = StandardScaler()` creates an empty brain. Once you `fit()` it, it remembers the math. If you type `scaler = StandardScaler()` again, you erase that memory. Reuse the exact same object for the test data.
* **Unseen Categorical Variables:** Real-world data will eventually throw an unseen category at you (like the title "Dona" in the test set). Always have a "catch-all" bucket (like 'Rare') to safely handle surprises without crashing the program.
* **The Final Sanity Check:** Before feeding data to a model, check `df.dtypes`. Every single column must be an `int` or a `float`. A single `object` (text string) will crash the model.
* **Model Persistence (Saving Models):** Traditional ML models (Scikit-Learn) are saved as `.pkl` (Pickle) or `.joblib` files. Deep Learning models (TensorFlow/Keras) are saved as `.h5` or `.keras` files.