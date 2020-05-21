# Simulated-Annealing-con-aplicaciones-en-Ciencias-de-Materiales

Tesis de Licenciatura en Matemática.
Disponible en: http://sedici.unlp.edu.ar/handle/10915/87857

Nuestro entendimiento de los fen´omenos naturales habitualmente se sobresimplifica en la capacidad de mensurar propiedades de un sistema de estudio, que luego se interpreta probando alguna
hip´otesis (usualmente formalizada en lenguaje matem´atico). Luego, esta hip´otesis se verifica a trav´es
de modelos que capturan su esencia o bien muta en nuevas hip´otesis.
En la ciencia de materiales, el uso de modelos param´etricos es de uso habitual para obtener informaci´on de mensurables f´ısicos. La informaci´on del mismo se obtiene al maximizar la verosimilitud
entre los datos experimentales y el modelo propuesto.
En el presente trabajo de tesis de grado para optar por el t´ıtulo de licenciado en matem´atica,
se propusieron y programaron modificaciones a algoritmos de Monte Carlo para el ajuste datos
experimentales empleando funciones param´etricas donde adem´as se analiz´o el rendimiento num´erico
de cada una de ellas. Se pusieron a prueba varias estrategias como la b´usqueda aleatoria simple,
m´etodos h´ıbridos basados en b´usqueda aleatoria combinados con algoritmos de m´aximo gradiente,
como as´ı tambi´en la implementaci´on de recocidos simulados (Simulated Annealing).
Entre los aportes originales de este trabajo de tesis, se destaca la propuesta de desacoplar los
par´ametros lineales de los no lineales. En esta metodolog´ıa, la b´usqueda aleatoria s´olo se centra
en los t´erminos no lineales y el resto de los t´erminos se computan en cada simulaci´on mediante
cuadrados m´ınimos lineales, a diferencia de metodolog´ıas convencionales de Monte Carlo donde la
exploraci´on se realiza sobre todos los par´ametros indistintamente de su naturaleza. Esta propuesta
disminuye el tiempo de m´aquina al reducir la dimensionalidad del espacio de muestreo.
