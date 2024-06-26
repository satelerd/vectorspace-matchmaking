import csv
import umap
import warnings
import seaborn as sns
from scipy import spatial
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
print()
print("Libraries imported successfully.")

# Paths
path = 'SmartVOC'
data_path = 'demo1'
img_path = 'demo1'

# Read data and their responses from a CSV file, replace data.csv with own link or file name
data_map = {}
gsc_map = {}
with open(f'./data/{path}/{data_path}.csv', newline='', encoding='utf-8') as csvfile:
    data = csv.reader(csvfile, delimiter=',', quotechar='"')
    next(data)  # Skip the header row
    for row in data:
        name, paragraph, gsc = row  # Asumimos que gsc es la tercera columna
        data_map[paragraph] = name
        gsc_map[name] = gsc
print("Data loaded successfully.")

# Generate sentence embeddings
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
paragraphs = list(data_map.keys())
embeddings = model.encode(paragraphs)
print("Embeddings generated successfully.")
    
# Create a dictionary to store embeddings for each person
person_embeddings = {data_map[paragraph]: embedding for paragraph, embedding in zip(paragraphs, embeddings)}
print(person_embeddings)
# Reducing dimensionality of embedding data, scaling to coordinate domain/range
reducer = umap.UMAP()
scaler = StandardScaler()
scaled_data = scaler.fit_transform(list(person_embeddings.values()))
reduced_data = reducer.fit_transform(scaled_data)
print("Dimensionality reduction and scaling completed successfully.")

# Creating lists of coordinates with accompanying labels
x = [row[0] for row in reduced_data]
y = [row[1] for row in reduced_data]
label = list(person_embeddings.keys())



def get_color_by_raw_interview(name):
    if "Investigador en Tecnología" in name:
        return "red"
    elif "Profesional Independiente en Diseño Gráfico" in name:
        return "green"
    elif "Gerente de Marketing en una Startup" in name:
        return "orange"
    elif "CEO de Retail" in name:
        return "purple"
    elif "Director de Finanzas de un Banco" in name:
        return "blue"
    elif "Profesional Independiente en Diseño Gráfico" in name:
        return "black"
    elif "Experto en Telecomunicaciones" in name:
        return "brown"
    elif "Analista de Datos" in name:
        return "cyan"
    elif "Desarrollador de IA" in name:
        return "magenta"
    elif "Consultor en Transformación Digital" in name:
        return "yellow"
    elif "Estudiante de Tecnología" in name:
        return "pink"
    elif "Empleado Administrativo" in name:
        return "lime"
    elif "Profesor Universitario" in name:
        return "teal"
    elif "Freelance de Contenidos" in name:
        return "navy"
    elif "Responsable de Recursos Humanos" in name:
        return "maroon"
    elif "Consultor de Estrategia Empresarial" in name:
        return "olive"
    elif "Director de Operaciones" in name:
        return "aqua"
    elif "Gerente de Finanzas" in name:
        return "silver"
    elif "Director de Marketing" in name:
        return "gold"
    elif "Gerente de Producto" in name:
        return "coral"
    elif "Responsable de Seguridad Informática" in name:
        return "khaki"
    elif "Jefe de Desarrollo de Negocios" in name:
        return "orchid"
    elif "Consultor de Tecnología" in name:
        return "plum"
    elif "Chief Data Officer" in name:
        return "salmon"
    elif "Responsable de Innovación" in name:
        return "violet"
    else:
        return "grey"  # Default color

def get_color_by_gsc(name):
    if "Positiva" in name:
        return "green"
    elif "Cauta" in name:
        return "red"
    elif "Escéptica" in name:
        return "blue"
    

    if "Eficiencia Operativa" in name:
        return "green"
    elif "Rapidez en Procesos" in name:
        return "blue"
    elif "Capacidad de Personalización" in name:
        return "orange"
    elif "Ahorro de Tiempo" in name:
        return "purple"
    elif "No Ventajas" in name:
        return "black"
    
    
    if "Reflexivo" in name:
        return "green"
    elif "Entusiasta" in name:
        return "blue"
    elif "Cauteloso" in name:
        return "yellow"
    

    else:
        return "grey"  # Default color
    

def get_color_by_gsc_demo(name):
    gsc = gsc_map.get(name, '').lower()  # Convertimos a minúsculas para hacer la comparación case-insensitive
    if "entusiasta" in gsc:
        return "#008000"  # Verde
    elif "positiva con reservas" in gsc:
        return "#00FF00"  # Lima
    elif "balanceada e interesado" in gsc:
        return "#FFFF00"  # Amarillo
    elif "neutral" in gsc:
        return "#FFA500"  # Naranja
    elif "cauteloso" in gsc:
        return "#FF0000"  # Rojo
    elif "escéptico" in gsc:
        return "#800080"  # Púrpura
    else: 
        return "#808080"  # Gris (por defecto)

# Define colors for each profile
def get_color(name):

    # return get_color_by_raw_interview(name)
    # return get_color_by_gsc(name)
    return get_color_by_gsc_demo(name)


# Assign colors to each label
colors = [get_color(name) for name in label]

print(x, y)
print("------")
print(colors)
# Plotting and annotating data points
plt.scatter(x, y, c=colors)
for i, name in enumerate(label):
    if "NPC" in name:
        plt.annotate(name, (x[i], y[i]), fontsize="0", color="black")
    else:
        plt.annotate(name[0:19], (x[i], y[i]), fontsize="0", color="black")  # Increased font size to 12 and set color to white

# Clean-up and Export
plt.axis('off')
plt.savefig(f'./imgs/{path}/{img_path}.png', dpi=1000)
print("Visualization saved successfully.")