import streamlit as st
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import logging
import os
from dataclasses import dataclass, asdict

# Configuração do logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# =============================================================================
# CLASSE PIECE – Representa uma peça a ser posicionada
# =============================================================================
@dataclass
class Piece:
    shape: str             # "rectangular" ou "circular"
    width: float = None    # Para peças retangulares
    height: float = None   # Para peças retangulares
    section: float = 0     # Para peças retangulares; se em branco, assume 0 (peça sólida)
    diameter: float = None # Para peças circulares (DE)
    inner_diameter: float = 0  # Para peças circulares (DI); se 0, peça sólida
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
        """
        Verifica e marca uma peça retangular (ou vazada) de dimensões (w x h).
        Usa slicing do NumPy para detectar se a área está livre.
        """
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
        """
        Busca posição livre na chapa para a peça retangular, varrendo em passos (linha e coluna).
        """
        step_x = max(1, int(margin + piece_width))
        step_y = max(1, int(margin + piece_height))
        # Varredura linha-primeiro
        for y in range(0, self.length - piece_height + 1, step_y):
            for x in range(0, self.width - piece_width + 1, step_x):
                if self.check_and_mark_rectangular(x, y, piece_width, piece_height, extra):
                    return (x, y)
        # Varredura coluna-primeiro
        for x in range(0, self.width - piece_width + 1, step_x):
            for y in range(0, self.length - piece_height + 1, step_y):
                if self.check_and_mark_rectangular(x, y, piece_width, piece_height, extra):
                    return (x, y)
        return None

    def check_and_mark_circular(self, x: int, y: int, diameter: int, inner_diameter: float) -> bool:
        """
        Verifica e marca uma peça circular (ou anelar) de diâmetro externo 'diameter'
        e diâmetro interno 'inner_diameter'. Implementado de forma vetorizada
        para melhorar a performance em peças grandes.
        """
        # Se ultrapassar as bordas da chapa, retorna False imediatamente
        if x + diameter > self.width or y + diameter > self.length:
            return False

        outer_radius = diameter / 2.0
        inner_radius = inner_diameter / 2.0

        # "Recorta" a região da chapa onde a peça seria colocada
        region = self.layout[y:y+diameter, x:x+diameter]

        # Cria índices para todas as posições [0..diameter-1, 0..diameter-1]
        idx = np.indices((diameter, diameter), dtype=float)
        # Calcula a distância de cada ponto ao centro da peça (outer_radius, outer_radius)
        dist = np.hypot((idx[0] + 0.5) - outer_radius, (idx[1] + 0.5) - outer_radius)

        if inner_radius > 0:
            # Máscara booleana para região anelar
            mask = (dist <= outer_radius) & (dist >= inner_radius)
        else:
            # Máscara booleana para região sólida
            mask = (dist <= outer_radius)

        # Verifica se todos os pontos onde mask=True estão livres (== 0)
        if not np.all(region[mask] == 0):
            return False

        # Marca a região na chapa
        region[mask] = 1
        return True

    def try_place_circular(self, diameter: int, inner_diameter: float, margin: int):
        """
        Busca posição livre na chapa para a peça circular, varrendo em passos.
        A varredura é reduzida para melhorar performance, mas ainda
        mantém a lógica de verificação.
        """
        # Heurística: se a peça é vazada (inner_diameter > 0), ou a margem é pequena,
        # reduzimos o passo para permitir encontrar espaços internos.
        # Caso contrário, usamos algo maior que a soma da margem + parte do diâmetro.
        if inner_diameter > 0:
            step = max(1, int(margin))  # busca mais fina para achar espaços internos
        else:
            # Usa 25% do diâmetro como passo adicional, evitando varreduras muito densas
            step = max(1, int(margin + diameter * 0.25))

        for y in range(0, self.length - diameter + 1, step):
            for x in range(0, self.width - diameter + 1, step):
                if self.check_and_mark_circular(x, y, diameter, inner_diameter):
                    return (x, y)

        # Se não encontrou com a varredura heurística, faz uma busca final completa
        # (ainda vetorizada, mas sem saltos). Isso garante não perder casos de encaixe apertado.
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
        """
        Executa a otimização de corte. A cada peça, tenta colocá-la na chapa atual.
        Se não couber, cria uma nova chapa (exceto em modo max_mode, que para ao encher a chapa).
        """
        # Para o progresso, considera-se 1 "tentativa" para max_mode e quantity para os demais.
        total_count = sum(piece.quantity if not piece.max_mode else 1 for piece in pieces)
        progress = 0

        for piece in pieces:
            if piece.max_mode:
                # Preenche somente a chapa atual com o máximo possível desta peça
                while True:
                    current_sheet = self._get_current_sheet()
                    placed = False
                    if piece.shape == "rectangular":
                        pos = current_sheet.try_place_rectangular(int(piece.width), int(piece.height),
                                                                  int(piece.section), int(self.margin))
                        orientation = "normal"
                        if pos is None and piece.rotatable:
                            pos = current_sheet.try_place_rectangular(int(piece.height), int(piece.width),
                                                                      int(piece.section), int(self.margin))
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
                    else:
                        # Circular
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
                # Modo "normal": respeita a quantidade de peças
                for _ in range(piece.quantity):
                    placed = False
                    while not placed:
                        current_sheet = self._get_current_sheet()
                        if piece.shape == "rectangular":
                            pos = current_sheet.try_place_rectangular(int(piece.width), int(piece.height),
                                                                      int(piece.section), int(self.margin))
                            orientation = "normal"
                            if pos is None and piece.rotatable:
                                pos = current_sheet.try_place_rectangular(int(piece.height), int(piece.width),
                                                                          int(piece.section), int(self.margin))
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
                                # Se não couber, cria nova chapa
                                self._add_new_sheet()

                        else:
                            # Circular
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
                                # Se não couber, cria nova chapa
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
                    extra = pos.get("section", 0)
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
                    if pos.get("inner_diameter", 0) > 0 and pos["inner_diameter"] < d:
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
        st.sidebar.subheader("Importar Peças via Excel/CSV")
        uploaded_file = st.sidebar.file_uploader("Escolha um arquivo (Excel ou CSV)", type=["xlsx", "xls", "csv"])
        if uploaded_file:
            try:
                # Verifica extensão do arquivo
                file_ext = os.path.splitext(uploaded_file.name)[1].lower()
                if file_ext in [".xlsx", ".xls"]:
                    df = pd.read_excel(uploaded_file)
                elif file_ext == ".csv":
                    df = pd.read_csv(uploaded_file)
                else:
                    st.sidebar.error("Formato não suportado. Selecione um arquivo Excel ou CSV.")
                    st.stop()

                df.columns = df.columns.str.strip()
                df.fillna("", inplace=True)

                # Colunas que esperamos encontrar
                required_cols = {"shape", "width", "height", "section", "diameter", "inner_diameter", "quantity", "rotatable", "max_mode"}
                if not required_cols.issubset(set(df.columns)):
                    st.sidebar.error(
                        f"As colunas do arquivo devem incluir: {', '.join(sorted(required_cols))}. "
                        f"Colunas encontradas: {', '.join(df.columns)}"
                    )
                else:
                    st.session_state["pieces"] = []
                    for idx, row in df.iterrows():
                        shape = str(row.get("shape", "")).strip().lower()
                        quantity = int(row.get("quantity", 1) or 1)
                        rotatable = bool(row.get("rotatable", True))
                        max_mode = bool(row.get("max_mode", False))

                        if shape == "rectangular":
                            w_val = float(row.get("width", 0) or 0)
                            h_val = float(row.get("height", 0) or 0)
                            s_val = float(row.get("section", 0) or 0)
                            new_piece = Piece(
                                shape="rectangular",
                                width=w_val,
                                height=h_val,
                                section=s_val,
                                quantity=quantity,
                                rotatable=rotatable,
                                max_mode=max_mode
                            )
                            st.session_state["pieces"].append(new_piece)

                        elif shape == "circular":
                            d_val = float(row.get("diameter", 0) or 0)
                            i_val = float(row.get("inner_diameter", 0) or 0)
                            new_piece = Piece(
                                shape="circular",
                                diameter=d_val,
                                inner_diameter=i_val,
                                quantity=quantity,
                                rotatable=rotatable,
                                max_mode=max_mode
                            )
                            st.session_state["pieces"].append(new_piece)
                        else:
                            st.sidebar.warning(f"Linha {idx+2}: 'shape' não reconhecido ({shape}). Ignorando peça.")

                    st.sidebar.success(f"Importação concluída com {len(st.session_state['pieces'])} peças.")
                    logging.info(f"Importação concluída. Total de peças: {len(st.session_state['pieces'])}")
            except Exception as e:
                st.sidebar.error(f"Erro ao ler o arquivo: {e}")

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
