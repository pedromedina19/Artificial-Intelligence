import tkinter as tk
from tkinter import messagebox
import regradehebb

# Crie a janela
window = tk.Tk()
window.title("Planilhas 8x8")

def on_button_click(button):
    current_text = button['text']
    new_text = '1' if current_text == ' ' else ' '
    button.config(text=new_text)

# Crie frames para as tabelas
frame1 = tk.Frame(window)
frame2 = tk.Frame(window)

# Defina as células das tabelas dentro dos frames
coluna1 = [[tk.Button(frame1, text=' ', width=3, command=lambda i=i, j=j: on_button_click(coluna1[i][j])) for j in range(8)] for i in range(8)]
coluna2 = [[tk.Button(frame2, text=' ', width=3, command=lambda i=i, j=j: on_button_click(coluna2[i][j])) for j in range(8)] for i in range(8)]

# Adicione as células das tabelas
for i in range(8):
    for j in range(8):
        coluna1[i][j].grid(row=i, column=j)
        coluna2[i][j].grid(row=i, column=j)

# Posicione os frames na janela
frame1.grid(row=0, column=0)
tk.Label(window, text="   ").grid(row=0, column=1)  # Espaço entre as tabelas
frame2.grid(row=0, column=2)

def pegar_Valores(colunas):
    valores_matrizes = []

    for tabela in colunas:
        for row in range(8):
            for column in range(8):
                valores_matrizes.append(
                    1 if tabela[row][column]['text'].strip() == '1' else -1)

    return valores_matrizes

# Defina os botões
def treinar():
    vetor_tabela1 = pegar_Valores([coluna1])
    vetor_tabela2 = pegar_Valores([coluna2])

    resultado = regradehebb.treinamentoHebb(vetor_tabela1, vetor_tabela2)
    messagebox.showinfo("Resultado", resultado + "\nPara testar use a Tabela 1")

def testar():
    vetor_tabela1 = pegar_Valores([coluna1])
    resultado = regradehebb.testeHebb(vetor_tabela1)
    messagebox.showinfo("Resultado", resultado)

def limpar():
    for row in range(8):
        for column in range(8):
            coluna1[row][column].config(text=' ')

tk.Button(window, text="Treinar", command=treinar).grid(row=1, column=0)
tk.Button(window, text="Testar", command=testar).grid(row=1, column=1)
tk.Button(window, text="Limpar Tabela 1", command=limpar).grid(row=2, column=0)

# Inicie o loop principal
window.mainloop()
