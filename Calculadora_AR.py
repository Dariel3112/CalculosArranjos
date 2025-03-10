import streamlit as st
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import logging
from dataclasses import dataclass, asdict

# Configuração do logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# =============================================================================
# CLASSE PIECE – Representa uma peça a ser posicionada
# =============================================================================
@dataclass
class Piece:
    # Para peças retangulares: use width, height e section.
    # Para peças circulares: use diameter (DE) e inner_diameter (DI).
    shape: str             # "rectangular" ou "circular"
    width: float = None    # para peças retangulares
    height: float = None   # para peças retangulares
    section: float = 0     # para peças retangulares; se em branco, assume 0 (peça sólida)
    diameter: float = None # para peças circulares (DE)
    inner_diameter: float = 0  # para peças circulares (DI); se 0, peça sólida
    quantity: int = 1
    rotatable: bool = True
    max_mode: bool = False

# =============================================================================
# CLASSE LAYOUTSHEET – Representa uma chapa única
# =============================================================================
class LayoutSheet:
    def __init__(self, width: float, length: float):
        self.width = int(width)
        self.length = int(length)
        self.layout = np.zeros((self.length, self.width), dtype=int)
        logging.info(f"Nova chapa criada: {self.width} x {self.length} mm")

    def check_and_mark_rectangular(self, x: int, y: int, w: int, h: int, extra: int) -> bool:
        if extra <= 0 or (extra * 2 >= w) or (extra * 2 >= h):
            region = self.layout[y:y+h, x:x+w]
            if np.all(region == 0):
                self.layout[y:y+h, x:x+w] = 1
                return True
            else:
                return False
        else:
            top = self.layout[y:y+extra, x:x+w]
            bottom = self.layout[y+h-extra:y+h, x:x+w]
            left = self.layout[y+extra:y+h-extra, x:x+extra]
            right = self.layout[y+extra:y+h-extra, x+w-extra:x+w]
            if np.all(top == 0) and np.all(bottom == 0) and np.all(left == 0) and np.all(right == 0):
                self.layout[y:y+extra, x:x+w] = 1
                self.layout[y+h-extra:y+h, x:x+w] = 1
                self.layout[y+extra:y+h-extra, x:x+extra] = 1
                self.layout[y+extra:y+h-extra, x+w-extra:x+w] = 1
                return True
            else:
                return False

    def try_place_rectangular(self, piece_width: int, piece_height: int, extra: int, margin: int):
        step = max(1, int(margin + piece_width))
        step_y = max(1, int(margin + piece_height))
        # Varredura linha-primeiro
        for y in range(0, self.length - piece_height + 1, step_y):
            for x in range(0, self.width - piece_width + 1, step):
                if self.check_and_mark_rectangular(x, y, piece_width, piece_height, extra):
                    return (x, y)
        # Varredura coluna-primeiro
        for x in range(0, self.width - piece_width + 1, step):
            for y in range(0, self.length - piece_height + 1, step_y):
                if self.check_and_mark_rectangular(x, y, piece_width, piece_height, extra):
                    return (x, y)
        return None

    def check_and_mark_circular(self, x: int, y: int, diameter: int, inner_diameter: float) -> bool:
        outer_radius = diameter / 2.0
        inner_radius = inner_diameter / 2.0
        cx = x + outer_radius
        cy = y + outer_radius
        free = True
        for i in range(y, y + diameter):
            for j in range(x, x + diameter):
                cell_center_x = j + 0.5
                cell_center_y = i + 0.5
                dist = math.hypot(cell_center_x - cx, cell_center_y - cy)
                if inner_diameter > 0:
                    if (dist <= outer_radius) and (dist >= inner_radius):
                        if self.layout[i, j] != 0:
                            free = False
                            break
                else:
                    if dist <= outer_radius:
                        if self.layout[i, j] != 0:
                            free = False
                            break
            if not free:
                break
        if free:
            for i in range(y, y + diameter):
                for j in range(x, x + diameter):
                    cell_center_x = j + 0.5
                    cell_center_y = i + 0.5
                    dist = math.hypot(cell_center_x - cx, cell_center_y - cy)
                    if inner_diameter > 0:
                        if (dist <= outer_radius) and (dist >= inner_radius):
                            self.layout[i, j] = 1
                    else:
                        if dist <= outer_radius:
                            self.layout[i, j] = 1
            return True
        else:
            return False

    def try_place_circular(self, diameter: int, inner_diameter: float, margin: int):
        # Se for peça vazada, utiliza step menor para detectar espaços internos livres
        if inner_diameter > 0:
            step = max(1, int(margin))
        else:
            step = max(1, int(margin + diameter))
        for y in range(0, self.length - diameter + 1, step):
            for x in range(0, self.width - diameter + 1, step):
                if self.check_and_mark_circular(x, y, diameter, inner_diameter):
                    return (x, y)
        # Busca completa, se não encontrada com step otimizado
        for y in range(0, self.length - diameter + 1):
            for x in range(0, self.width - diameter + 1):
                if self.check_and_mark_circular(x, y, diameter, inner_diameter):
                    return (x, y)
        return None

# =============================================================================
# CLASSE CUTOPTIMIZER – Responsável pela otimização dos cortes
# =============================================================================
class CutOptimizer:
    def __init__(self, sheet_width: float, sheet_length: float, margin: float):
        self.sheet_width = sheet_width
        self.sheet_length = sheet_length
        self.margin = margin
        self.sheets = []
        self.sheets.append(LayoutSheet(sheet_width, sheet_length))
        self.positions = []
        self.total_pieces = 0

    def _get_current_sheet(self):
        return self.sheets[-1]

    def _add_new_sheet(self):
        new_sheet = LayoutSheet(self.sheet_width, self.sheet_length)
        self.sheets.append(new_sheet)
        logging.info(f"Nova chapa adicionada. Total de chapas: {len(self.sheets)}")

    def optimize(self, pieces: list, progress_callback=None):
        # Para o progresso, considera-se 1 "tentativa" para max_mode e quantidade para os demais.
        total_count = sum(piece.quantity if not piece.max_mode else 1 for piece in pieces)
        progress = 0
        for piece in pieces:
            if piece.max_mode:
                # Preenche somente a chapa atual com o máximo possível desta peça
                while True:
                    current_sheet = self._get_current_sheet()
                    placed = False
                    if piece.shape == "rectangular":
                        pos = current_sheet.try_place_rectangular(int(piece.width), int(piece.height), int(piece.section), int(self.margin))
                        orientation = "normal"
                        if pos is None and piece.rotatable:
                            pos = current_sheet.try_place_rectangular(int(piece.height), int(piece.width), int(piece.section), int(self.margin))
                            orientation = "rotated" if pos is not None else None
                        if pos is not None:
                            self.positions.append({
                                "sheet": len(self.sheets),
                                "x": pos[0],
                                "y": pos[1],
                                "width": piece.width if orientation == "normal" else piece.height,
                                "height": piece.height if orientation == "normal" else piece.width,
                                "shape": "rectangular",
                                "section": piece.section,
                                "orientation": orientation
                            })
                            self.total_pieces += 1
                            placed = True
                            logging.info(f"Peça retangular (max_mode) posicionada na chapa {len(self.sheets)} em {pos} com orientação {orientation}")
                    elif piece.shape == "circular":
                        pos = current_sheet.try_place_circular(int(piece.diameter), piece.inner_diameter, int(self.margin))
                        if pos is not None:
                            self.positions.append({
                                "sheet": len(self.sheets),
                                "x": pos[0],
                                "y": pos[1],
                                "diameter": piece.diameter,
                                "inner_diameter": piece.inner_diameter,
                                "shape": "circular",
                                "orientation": "normal"
                            })
                            self.total_pieces += 1
                            placed = True
                            logging.info(f"Peça circular (max_mode) posicionada na chapa {len(self.sheets)} em {pos}")
                    if not placed:
                        # Se não couber mais na chapa atual, encerra max_mode (não cria nova chapa para max_mode)
                        break
                    progress += 1
                    if progress_callback:
                        progress_callback(min(progress / total_count, 1.0))
            else:
                for _ in range(piece.quantity):
                    placed = False
                    while not placed:
                        current_sheet = self._get_current_sheet()
                        if piece.shape == "rectangular":
                            pos = current_sheet.try_place_rectangular(int(piece.width), int(piece.height), int(piece.section), int(self.margin))
                            orientation = "normal"
                            if pos is None and piece.rotatable:
                                pos = current_sheet.try_place_rectangular(int(piece.height), int(piece.width), int(piece.section), int(self.margin))
                                orientation = "rotated" if pos is not None else None
                            if pos is not None:
                                self.positions.append({
                                    "sheet": len(self.sheets),
                                    "x": pos[0],
                                    "y": pos[1],
                                    "width": piece.width if orientation == "normal" else piece.height,
                                    "height": piece.height if orientation == "normal" else piece.width,
                                    "shape": "rectangular",
                                    "section": piece.section,
                                    "orientation": orientation
                                })
                                self.total_pieces += 1
                                placed = True
                                logging.info(f"Peça retangular posicionada na chapa {len(self.sheets)} em {pos} com orientação {orientation}")
                            else:
                                self._add_new_sheet()
                        elif piece.shape == "circular":
                            pos = current_sheet.try_place_circular(int(piece.diameter), piece.inner_diameter, int(self.margin))
                            if pos is not None:
                                self.positions.append({
                                    "sheet": len(self.sheets),
                                    "x": pos[0],
                                    "y": pos[1],
                                    "diameter": piece.diameter,
                                    "inner_diameter": piece.inner_diameter,
                                    "shape": "circular",
                                    "orientation": "normal"
                                })
                                self.total_pieces += 1
                                placed = True
                                logging.info(f"Peça circular posicionada na chapa {len(self.sheets)} em {pos}")
                            else:
                                self._add_new_sheet()
                    progress += 1
                    if progress_callback:
                        progress_callback(min(progress / total_count, 1.0))
        return self.positions, self.total_pieces, len(self.sheets)

# =============================================================================
# FUNÇÃO PARA PLOTAR O LAYOUT (utilizando matplotlib)
# =============================================================================
def plot_layout(sheet_width: float, sheet_length: float, positions: list, total_pieces: int, needed_sheets: int):
    if needed_sheets == 1:
        fig, ax = plt.subplots(figsize=(6, 6))
        sheets_axes = [ax]
    else:
        fig, axes = plt.subplots(1, needed_sheets, figsize=(6 * needed_sheets, 6))
        sheets_axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]

    for i in range(1, needed_sheets + 1):
        ax = sheets_axes[i - 1]
        ax.set_title(f"Chapa {i}")
        ax.set_xlim(0, sheet_width)
        ax.set_ylim(0, sheet_length)
        ax.set_aspect('equal')
        sheet_rect = plt.Rectangle((0, 0), sheet_width, sheet_length, fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(sheet_rect)
        for pos in positions:
            if pos["sheet"] == i:
                if pos["shape"] == "rectangular":
                    x, y = pos["x"], pos["y"]
                    w = pos["width"]
                    h = pos["height"]
                    extra = pos["section"]
                    if extra > 0 and (2 * extra < w) and (2 * extra < h):
                        rect_outer = plt.Rectangle((x, y), w, h, edgecolor='blue', facecolor='cyan', linewidth=1)
                        ax.add_patch(rect_outer)
                        inner_rect = plt.Rectangle((x + extra, y + extra), w - 2 * extra, h - 2 * extra,
                                                   edgecolor='red', facecolor='none', linestyle='dashed', linewidth=1)
                        ax.add_patch(inner_rect)
                    else:
                        rect_patch = plt.Rectangle((x, y), w, h, edgecolor='blue', facecolor='cyan', linewidth=1)
                        ax.add_patch(rect_patch)
                elif pos["shape"] == "circular":
                    x, y = pos["x"], pos["y"]
                    d = pos["diameter"]
                    center = (x + d / 2, y + d / 2)
                    circle = plt.Circle(center, d / 2, edgecolor='green', facecolor='lightgreen', linewidth=1)
                    ax.add_patch(circle)
                    if pos["inner_diameter"] > 0 and pos["inner_diameter"] < d:
                        inner_circle = plt.Circle(center, pos["inner_diameter"] / 2,
                                                  edgecolor='red', facecolor='none', linestyle='dashed', linewidth=1)
                        ax.add_patch(inner_circle)
        ax.invert_yaxis()
    plt.tight_layout()
    return fig

# =============================================================================
# INTERFACE COM STREAMLIT
# =============================================================================
def main():
    st.title("Sistema de Otimização de Corte")
    st.write("Interface moderna com Streamlit e lógica desacoplada para facilitar futuras migrações.")

    # Inicializa a lista de peças na session_state (mantendo os dados entre interações)
    if "pieces" not in st.session_state:
        st.session_state["pieces"] = []

    # Seleção do modo de inserção: Manual ou Excel
    mode = st.sidebar.radio("Selecione o modo de inserção de peças", ("Manual", "Excel"))

    # Botão para limpar a lista de peças
    if st.sidebar.button("Limpar lista de peças"):
        st.session_state["pieces"] = []
        st.sidebar.success("Lista de peças limpa.")

    # Opção para excluir peça
    if st.session_state["pieces"]:
        st.sidebar.subheader("Excluir Peça")
        pieces_str = [f"{i}: " +
                      (f"Retangular: {p.width}x{p.height} mm, Seção: {p.section}, Qtd: {p.quantity}" if p.shape == "rectangular"
                       else f"Circular: DE {p.diameter} mm, DI {p.inner_diameter} mm, Qtd: {p.quantity}")
                      for i, p in enumerate(st.session_state["pieces"])]
        selected_index = st.sidebar.selectbox("Selecione a peça para excluir", options=range(len(pieces_str)), format_func=lambda i: pieces_str[i])
        if st.sidebar.button("Excluir Peça"):
            st.session_state["pieces"].pop(selected_index)
            st.sidebar.success("Peça excluída com sucesso!")

    # =============================================================================
    # MODO MANUAL
    # =============================================================================
    if mode == "Manual":
        st.sidebar.subheader("Adicionar Peça Manualmente")
        piece_type = st.sidebar.radio("Formato da Peça", ("Retangular", "Circular"), key="piece_type")
        if piece_type == "Retangular":
            width = st.sidebar.text_input("Largura (mm)", value="50", key="width")
            height = st.sidebar.text_input("Comprimento (mm)", value="80", key="height")
            section = st.sidebar.text_input("Seção (mm) [opcional, deixe em branco para peça sólida]", value="5", key="section")
            quantity = st.sidebar.text_input("Quantidade", value="1", key="quantity_rect")
            rotatable = st.sidebar.checkbox("Permitir Rotação", value=True, key="rotatable")
            max_mode = st.sidebar.checkbox("Calcular máximo na chapa", value=False, key="max_mode_rect")
            if st.sidebar.button("Adicionar Peça Retangular"):
                try:
                    w_val = float(width)
                    h_val = float(height)
                    sec_val = float(section) if section.strip() != "" else 0
                    qty_val = int(quantity) if quantity.strip() != "" else 1
                    new_piece = Piece(
                        shape="rectangular",
                        width=w_val,
                        height=h_val,
                        section=sec_val,
                        quantity=qty_val,
                        rotatable=rotatable,
                        max_mode=max_mode
                    )
                    st.session_state["pieces"].append(new_piece)
                    st.sidebar.success("Peça retangular adicionada com sucesso!")
                    logging.info(f"Peça retangular adicionada: {asdict(new_piece)}")
                except ValueError as e:
                    st.sidebar.error(f"Erro na entrada de dados da peça retangular: {e}")
        else:
            diameter = st.sidebar.text_input("DE (mm)", value="60", key="diameter")
            inner_diameter = st.sidebar.text_input("DI (mm) [opcional, deixe em branco para peça sólida]", value="0", key="inner_diameter")
            quantity = st.sidebar.text_input("Quantidade", value="1", key="quantity_circ")
            max_mode = st.sidebar.checkbox("Calcular máximo na chapa", value=False, key="max_mode_circ")
            if st.sidebar.button("Adicionar Peça Circular"):
                try:
                    d_val = float(diameter)
                    inner_val = float(inner_diameter) if inner_diameter.strip() != "" else 0
                    qty_val = int(quantity) if quantity.strip() != "" else 1
                    new_piece = Piece(
                        shape="circular",
                        diameter=d_val,
                        inner_diameter=inner_val,
                        quantity=qty_val,
                        rotatable=True,
                        max_mode=max_mode
                    )
                    st.session_state["pieces"].append(new_piece)
                    st.sidebar.success("Peça circular adicionada com sucesso!")
                    logging.info(f"Peça circular adicionada: {asdict(new_piece)}")
                except ValueError as e:
                    st.sidebar.error(f"Erro na entrada de dados da peça circular: {e}")
    # =============================================================================
    # MODO EXCEL
    # =============================================================================
    else:
        st.sidebar.subheader("Importar Peças via Excel")
        uploaded_file = st.sidebar.file_uploader("Escolha um arquivo Excel", type=["xlsx", "xls"])
        if uploaded_file:
            try:
                df = pd.read_excel(uploaded_file)
                df.columns = df.columns.str.strip()
                df.fillna("", inplace=True)
                required_columns = {"Formato", "L", "C", "S", "Quantidade"}
                if not required_columns.issubset(set(df.columns)):
                    st.sidebar.error(f"O arquivo deve conter as colunas: {', '.join(required_columns)}")
                else:
                    st.session_state["pieces"] = []
                    for idx, row in df.iterrows():
                        formato = str(row.get("Formato", "")).strip().capitalize()
                        if not formato:
                            continue
                        try:
                            qty = int(row.get("Quantidade", 1))
                        except Exception:
                            qty = 1
                        if formato.lower().startswith("reta") or formato.lower().startswith("rect"):
                            try:
                                w_val = float(row.get("L", ""))
                                h_val = float(row.get("C", ""))
                                sec_val = float(row.get("S", "0")) if str(row.get("S", "")).strip() != "" else 0
                                new_piece = Piece(
                                    shape="rectangular",
                                    width=w_val,
                                    height=h_val,
                                    section=sec_val,
                                    quantity=qty,
                                    rotatable=True,
                                    max_mode=False
                                )
                                st.session_state["pieces"].append(new_piece)
                            except Exception as e:
                                st.sidebar.error(f"Erro na conversão dos dados da peça retangular na linha {idx+2}: {e}")
                        elif formato.lower().startswith("cir"):
                            try:
                                d_val = float(row.get("L", ""))
                                inner_val = float(row.get("C", "0")) if str(row.get("C", "")).strip() != "" else 0
                                new_piece = Piece(
                                    shape="circular",
                                    diameter=d_val,
                                    inner_diameter=inner_val,
                                    quantity=qty,
                                    rotatable=True,
                                    max_mode=False
                                )
                                st.session_state["pieces"].append(new_piece)
                            except Exception as e:
                                st.sidebar.error(f"Erro na conversão dos dados da peça circular na linha {idx+2}: {e}")
                    st.sidebar.success(f"Importação concluída com {len(st.session_state['pieces'])} peças.")
                    logging.info(f"Importação via Excel concluída com {len(st.session_state['pieces'])} peças.")
            except Exception as e:
                st.sidebar.error(f"Erro ao ler o arquivo Excel: {e}")

    # Exibe a lista de peças cadastradas
    if st.session_state["pieces"]:
        st.subheader("Peças Cadastradas")
        df_pieces = pd.DataFrame([asdict(p) for p in st.session_state["pieces"]])
        st.dataframe(df_pieces)

    # =============================================================================
    # PARÂMETROS DA CHAPA E EXECUÇÃO DA OTIMIZAÇÃO
    # =============================================================================
    st.header("Parâmetros da Chapa")
    sheet_width_input = st.text_input("Largura da Chapa (mm)", value="300", key="sheet_width")
    sheet_length_input = st.text_input("Comprimento da Chapa (mm)", value="400", key="sheet_length")
    margin_input = st.text_input("Margem (mm)", value="5", key="margin")

    if st.button("Executar Otimização"):
        try:
            sheet_width = float(sheet_width_input)
            sheet_length = float(sheet_length_input)
            margin = float(margin_input)
            if margin < 0:
                st.error("A margem não pode ser negativa.")
                return
        except ValueError as e:
            st.error(f"Erro nos parâmetros da chapa: {e}")
            logging.error(f"Erro de conversão dos parâmetros da chapa: {e}")
            return

        if not st.session_state["pieces"]:
            st.error("Nenhuma peça cadastrada. Adicione peças manualmente ou via Excel.")
            return

        progress_bar = st.progress(0)
        def update_progress(prog):
            progress_bar.progress(min(int(prog * 100), 100))

        optimizer = CutOptimizer(sheet_width, sheet_length, margin)
        positions, total_pieces, needed_sheets = optimizer.optimize(st.session_state["pieces"], progress_callback=update_progress)
        st.success("Otimização concluída!")
        st.write(f"Total de peças posicionadas: **{total_pieces}**")
        st.write(f"Chapas utilizadas: **{needed_sheets}**")
        fig = plot_layout(sheet_width, sheet_length, positions, total_pieces, needed_sheets)
        st.pyplot(fig)
        st.subheader("Posições das Peças")
        st.dataframe(pd.DataFrame(positions))

if __name__ == "__main__":
    main()
