# Simulated-Annealing-con-aplicaciones-en-Ciencias-de-Materiales

Tesis de Licenciatura en Matemática.
Disponible en: http://sedici.unlp.edu.ar/handle/10915/87857

Nuestro entendimiento de los fenómenos naturales habitualmente se sobresimplifica en la capacidad de mensurar propiedades de un sistema de estudio, que luego se interpreta probando alguna hipótesis (usualmente formalizada en lenguaje matemático). Luego, esta hipótesis se verifica a través de modelos que capturan su esencia o bien muta en nuevas hipótesis.

En la ciencia de materiales, el uso de modelos paramétricos es de uso habitual para obtener información de mensurables físicos. La información del mismo se obtiene al maximizar la verosimilitud entre los datos experimentales y el modelo propuesto.
En el presente trabajo de tesis de grado para optar por el título de licenciado en matemática, se propusieron y programaron modificaciones a algoritmos de Monte Carlo para el ajuste datos experimentales empleando funciones paramétricas donde además se analizó el rendimiento numérico de cada una de ellas. Se pusieron a prueba varias estrategias como la búsqueda aleatoria simple, métodos híbridos basados en búsqueda aleatoria combinados con algoritmos de máximo gradiente, como así también la implementación de recocidos simulados (Simulated Annealing).

Entre los aportes originales de este trabajo de tesis, se destaca la propuesta de desacoplar los parámetros lineales de los no lineales. En esta metodología, la búsqueda aleatoria sólo se centra en los términos no lineales y el resto de los términos se computan en cada simulación mediante cuadrados mínimos lineales, a diferencia de metodologías convencionales de Monte Carlo donde la exploración se realiza sobre todos los parámetros, indistintamente de su naturaleza. Esta propuesta disminuye el tiempo de máquina al reducir la dimensionalidad del espacio de muestreo.
