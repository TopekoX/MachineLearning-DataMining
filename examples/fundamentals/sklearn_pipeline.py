from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from customer_data import make_customer_data

SEED = 42
df = make_customer_data().drop_duplicates().reset_index(drop=True)
X, y = df.drop(columns="churn"), df["churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

numeric = ["age", "tenure_months", "income", "n_orders", "avg_spend"]
categorical = ["city", "plan"]

# Praproses berbeda untuk kolom numerik dan kategorik
preprocess = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                      ("scale", StandardScaler())]), numeric),
    ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
])

model = Pipeline([("preprocess", preprocess),
                  ("clf", LogisticRegression(max_iter=1000))])

# Validasi silang: setiap lipatan melatih ulang SELURUH pipeline pada data latih lipatan itu
scores = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc")
print("ROC-AUC CV: %.3f (+/- %.3f)" % (scores.mean(), scores.std()))

model.fit(X_train, y_train)
print("Akurasi uji : %.3f" % model.score(X_test, y_test))

names = model.named_steps["preprocess"].get_feature_names_out()
print("jumlah fitur setelah praproses:", len(names))
print(list(names))
print("langkah pipeline:", list(model.named_steps))
