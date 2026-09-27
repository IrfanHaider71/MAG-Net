
from IPython.display import display, Math

operator_formulas = [
    ("First Zagreb (M₁)",      r"(D_x + D_y)\,M(G;x,y)\big|_{x=y=1}"),
    ("Second Zagreb (M₂)",     r"(D_x D_y)\,M(G;x,y)\big|_{x=y=1}"),
    ("Modified second (mM₂)",  r"(S_x S_y)\,M(G;x,y)\big|_{x=y=1} \;=\; \sum_{ij}\frac{m_{ij}}{ij}"),
    ("Redefined M₃ (ReM₃)",    r"(D_x D_y (D_x + D_y))\,M(G;x,y)\big|_{x=y=1}"),
    ("Harmonic (H)",           r"\big(2\,S_x\,J\big)\,M(G;x,y)\big|_{x=1} \;=\; \sum_{ij}\frac{2\,m_{ij}}{i+j}"),
    ("Forgotten (F)",          r"\big(D_x^2 + D_y^2\big)\,M(G;x,y)\big|_{x=y=1}"),
    ("Augmented Zagreb (AZ)",  r"\left(S_x^{3}\,Q_{-2}\,J\,D_x^{3}\,D_y^{3}\right)\,M(G;x,y)\big|_{x=1}"),
    ("Inverse sum (ISI)",      r"\left(S_x\,J\,D_x\,D_y\right)\,M(G;x,y)\big|_{x=1} \;=\; \sum_{ij}\frac{ij}{i+j}\,m_{ij}"),
    ("Symmetric division (SDD)",r"\left(D_x S_y + D_y S_x\right)\,M(G;x,y)\big|_{x=y=1} \;=\; \sum_{ij}\left(\frac{i}{j}+\frac{j}{i}\right)m_{ij}"),
]

for name, latex in operator_formulas:
    print(name)
    display(Math(latex))
direct_defs = [
    ("M₁",  r"\sum_{uv\in E}\big(d(u)+d(v)\big)"),
    ("M₂",  r"\sum_{uv\in E} d(u)\,d(v)"),
    ("mM₂", r"\sum_{uv\in E} \frac{1}{d(u)\,d(v)}"),
    ("ReM₃",r"\sum_{uv\in E} d(u)\,d(v)\,\big(d(u)+d(v)\big)"),
    ("H",   r"\sum_{uv\in E} \frac{2}{d(u)+d(v)}"),
    ("F",   r"\sum_{uv\in E} \big(d(u)^2 + d(v)^2\big) \;=\; \sum_{v\in V} d(v)^3"),
    ("AZ",  r"\sum_{uv\in E} \left(\frac{d(u)\,d(v)}{d(u)+d(v)-2}\right)^3"),
    ("ISI", r"\sum_{uv\in E} \frac{d(u)\,d(v)}{d(u)+d(v)}"),
    ("SDD", r"\sum_{uv\in E} \left(\frac{d(u)}{d(v)}+\frac{d(v)}{d(u)}\right)"),
]
for name, latex in direct_defs:
    print(name + " (edge-sum form)")
    display(Math(latex))
