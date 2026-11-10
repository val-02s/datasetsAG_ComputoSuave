import random
import warnings
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, log_loss

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
POBLACION = 14
GENERACIONES = 10
PROB_MUTACION = 0.2
random.seed(RANDOM_STATE)

# datos
datos = load_breast_cancer(as_frame=True)
X = datos.data
y = datos.target

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=RANDOM_STATE, stratify=y_temp
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


def crear_red(ind):
    return MLPClassifier(
        hidden_layer_sizes=(ind["neuronas"],) * ind["capas"],
        activation=ind["activacion"],
        solver="adam",
        learning_rate_init=ind["lr"],
        alpha=ind["alpha"],
        batch_size=32,
        max_iter=300,
        random_state=RANDOM_STATE
    )


def mostrar(ind):
    red = crear_red(ind).fit(X_train, y_train)
    prob = red.predict_proba(X_test)
    print("Parametros:", ind)
    print("Exactitud prueba:", round(accuracy_score(y_test, prob.argmax(axis=1)), 4))
    print("Perdida prueba:", round(log_loss(y_test, prob), 4))
    print("Perdida entrenamiento:", round(red.loss_, 4))


# red inicial
inicial = {"capas": 1, "neuronas": 8, "lr": 0.001, "alpha": 0.0001, "activacion": "relu"}
print("\nRed inicial")
mostrar(inicial)


# algoritmo genetico
def crear_individuo():
    return {
        "capas": random.randint(1, 3),
        "neuronas": random.randint(2, 64),
        "lr": 10 ** random.uniform(-4, -1),
        "alpha": 10 ** random.uniform(-6, -1),
        "activacion": random.choice(["relu", "tanh", "logistic"]),
    }


def aptitud(ind):
    red = crear_red(ind).fit(X_train, y_train)
    prob = red.predict_proba(X_val)
    return accuracy_score(y_val, prob.argmax(axis=1)) - 0.05 * log_loss(y_val, prob)


def seleccion(poblacion, aptitudes):
    candidatos = random.sample(range(len(poblacion)), 3)
    mejor = max(candidatos, key=lambda i: aptitudes[i])
    return poblacion[mejor]


def cruce(padre1, padre2):
    return {k: random.choice([padre1[k], padre2[k]]) for k in padre1}


def mutacion(ind):
    nuevo = crear_individuo()
    return {k: nuevo[k] if random.random() < PROB_MUTACION else ind[k] for k in ind}


poblacion = [crear_individuo() for _ in range(POBLACION)]

for gen in range(GENERACIONES):
    aptitudes = [aptitud(ind) for ind in poblacion]
    orden = sorted(range(POBLACION), key=lambda i: aptitudes[i], reverse=True)
    poblacion = [poblacion[i] for i in orden]
    aptitudes = [aptitudes[i] for i in orden]
    print(f"\nGeneracion {gen}")
    print("Mejor aptitud:", round(aptitudes[0], 4))

    nueva = poblacion[:2]
    while len(nueva) < POBLACION:
        hijo = cruce(seleccion(poblacion, aptitudes), seleccion(poblacion, aptitudes))
        nueva.append(mutacion(hijo))
    poblacion = nueva

mejor = max(poblacion, key=aptitud)
print("\nRed optimizada")
mostrar(mejor)