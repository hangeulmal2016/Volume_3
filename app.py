import streamlit as st
import numpy as np
import pandas as pd
import ezdxf
from ezdxf.enums import TextEntityAlignment
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, MultiPolygon, Point
import io

st.set_page_config(page_title="Earthwork Grid Calculator", layout="wide")
st.title("🧮 Web App Tính Khối Lượng Đào Đắp Tùy Chọn Ranh Giới Nâng Cao")

# --- KHỞI TẠO TRẠNG THÁI LƯU TRỮ (SESSION STATE) ---
if "calculated" not in st.session_state:
    st.session_state.calculated = False
if "df_result" not in st.session_state:
    st.session_state.df_result = None
if "total_cut" not in st.session_state:
    st.session_state.total_cut = 0.0
if "total_fill" not in st.session_state:
    st.session_state.total_fill = 0.0
if "cad_grid_data" not in st.session_state:
    st.session_state.cad_grid_data = []
if "pts1_real" not in st.session_state:
    st.session_state.pts1_real = None
if "pts2_real" not in st.session_state:
    st.session_state.pts2_real = None
if "boundary_poly_coords" not in st.session_state:
    st.session_state.boundary_poly_coords = None
if "boundary_source" not in st.session_state:
    st.session_state.boundary_source = "surface2"

# --- GIAO DIỆN NHẬP LIỆU (SIDEBAR) ---
st.sidebar.header("1. Cấu hình Dữ liệu Đầu vào")

def parse_surface_input(label):
    st.sidebar.subheader(f"Bề mặt {label}")
    mode = st.sidebar.selectbox(f"Loại dữ liệu Bề mặt {label}", ["File TXT", "Giá trị Cao độ cố định (Mặt phẳng)"], key=f"mode_{label}")
    
    if mode == "Giá trị Cao độ cố định (Mặt phẳng)":
        z_val = st.sidebar.number_input(f"Nhập cao độ hằng số cho Bề mặt {label}", value=0.0, key=f"z_{label}")
        return {"type": "const", "value": z_val}
    else:
        file = st.sidebar.file_uploader(f"Tải lên file TXT Bề mặt {label} (Định dạng: X Y Z)", type=["txt"], key=f"file_txt_{label}")
        return {"type": "txt", "value": file}

surface_1 = parse_surface_input("1 (Hiện trạng)")
surface_2 = parse_surface_input("2 (Thiết kế)")

st.sidebar.subheader("2. Cấu hình Ranh giới")
boundary_mode = st.sidebar.selectbox(
    "Chọn nguồn dữ liệu ranh giới", 
    ["Sử dụng chu vi bề mặt làm ranh giới", "Tải lên file TXT Ranh giới", "Tải lên file DXF Ranh giới"]
)

sub_boundary_mode = None
if boundary_mode == "Sử dụng chu vi bề mặt làm ranh giới":
    sub_boundary_mode = st.sidebar.radio(
        "Tùy chọn thuật toán chu vi:",
        ["Sử dụng chu vi bề mặt 2", "Sử dụng chu vi từng bề mặt (Vùng giao nhau)"]
    )

boundary_file = None
if boundary_mode != "Sử dụng chu vi bề mặt làm ranh giới":
    boundary_file = st.sidebar.file_uploader("Tải lên file ranh giới", type=["txt", "dxf"], key="boundary_file_upload")

grid_size = st.sidebar.number_input("Kích thước cạnh ô lưới vuông (m)", min_value=1.0, value=5.0, step=1.0)

# --- CÁC HÀM PARSER ĐỌC DỮ LIỆU THỰC TẾ ---
def load_real_points(surface_dict):
    if surface_dict["type"] == "const" or surface_dict["value"] is None:
        return None
    points = []
    try:
        content = surface_dict["value"].read().decode("utf-8")
        surface_dict["value"].seek(0)
        for line in content.strip().split("\n"):
            line = line.strip()
            if not line: continue
            cleaned_line = line.replace(",", " ")
            parts = cleaned_line.split()
            if len(parts) >= 3:
                points.append([float(parts[0]), float(parts[1]), float(parts[2])])
    except:
        return None
    return np.array(points) if len(points) > 0 else None
def parse_boundary(mode, sub_mode, file_obj, pts1, pts2):
    if mode == "Sử dụng chu vi bề mặt làm ranh giới":
        if sub_mode == "Sử dụng chu vi bề mặt 2":
            if pts2 is None: return None, "surface2"
            hull2 = ConvexHull(pts2[:, :2])
            return Polygon(pts2[hull2.vertices, :2]), "surface2"
        elif sub_mode == "Sử dụng chu vi từng bề mặt (Vùng giao nhau)":
            if pts1 is None or pts2 is None: return None, "surfaces_intersect"
            hull1 = ConvexHull(pts1[:, :2])
            hull2 = ConvexHull(pts2[:, :2])
            poly1 = Polygon(pts1[hull1.vertices, :2])
            poly2 = Polygon(pts2[hull2.vertices, :2])
            if poly1.intersects(poly2):
                intersect_poly = poly1.intersection(poly2)
                if isinstance(intersect_poly, Polygon):
                    return intersect_poly, "surfaces_intersect"
            return None, "surfaces_intersect"
    if file_obj is None: return None, "custom"
    if "TXT" in mode:
        coords = []
        try:
            content = file_obj.read().decode("utf-8")
            file_obj.seek(0)
            for line in content.strip().split("\n"):
                line = line.strip()
                if not line: continue
                cleaned_line = line.replace(",", " ")
                parts = cleaned_line.split()
                if len(parts) >= 2:
                    coords.append((float(parts[0]), float(parts[1])))
            return Polygon(coords) if len(coords) >= 3 else None, "custom"
        except: return None, "custom"
    if "DXF" in mode:
        try:
            dxf_stream = io.BytesIO(file_obj.read())
            file_obj.seek(0)
            doc = ezdxf.read(dxf_stream)
            msp = doc.modelspace()
            for entity in msp.query('LWPOLYLINE POLYLINE'):
                coords = [pt[:2] for pt in entity.points()]
                if len(coords) >= 3:
                    return Polygon(coords), "custom"
        except: return None, "custom"
    return None, "custom"

# --- XỬ LÝ TÍNH TOÁN KHI NHẤN NÚT ---
if st.sidebar.button("👉 Tiến hành tính toán khối lượng"):
    pts1 = load_real_points(surface_1)
    pts2 = load_real_points(surface_2)
    st.session_state.pts1_real = pts1
    st.session_state.pts2_real = pts2
    
    boundary_polygon, b_source = parse_boundary(boundary_mode, sub_boundary_mode, boundary_file, pts1, pts2)
    st.session_state.boundary_source = b_source
    
    valid = True
    if surface_1["type"] == "txt" and pts1 is None:
        st.sidebar.error("❌ Kiểm tra lại file TXT Bề mặt 1.")
        valid = False
    if surface_2["type"] == "txt" and pts2 is None:
        st.sidebar.error("❌ Kiểm tra lại file TXT Bề mặt 2.")
        valid = False
    if boundary_polygon is None:
        st.sidebar.error("❌ Không thể khởi tạo ranh giới hợp lệ.")
        valid = False
        
    if valid:
        st.session_state.boundary_poly_coords = list(boundary_polygon.exterior.coords)
        x_min, y_min, x_max, y_max = boundary_polygon.bounds
        
        x_coords = np.arange(x_min, x_max + grid_size, grid_size)
        y_coords = np.arange(y_min, y_max + grid_size, grid_size)
        
        grid_rows_list = []
        cad_cells = []
        total_cut_vol = 0.0
        total_fill_vol = 0.0
        
        # ĐỊNH NGHĨA KÍCH THƯỚC LƯỚI VI PHÂN SIÊU MỊN ĐỂ TRIỆT TIÊU SAI SỐ HÌNH HỌC (0.5m)
        SUB_STEP = 0.5 
        
        for r_idx in range(len(y_coords) - 1):
            row_cells_data = []
            y_start = y_coords[r_idx]
            y_end = y_coords[r_idx + 1]
            
            for c_idx in range(len(x_coords) - 1):
                x_start = x_coords[c_idx]
                x_end = x_coords[c_idx + 1]
                
                cell_poly = Polygon([(x_start, y_start), (x_end, y_start), (x_end, y_end), (x_start, y_end)])
                if not cell_poly.intersects(boundary_polygon):
                    row_cells_data.append("Ngoài RG")
                    continue
                
                intersected_geo = cell_poly.intersection(boundary_polygon)
                actual_area = intersected_geo.area
                if actual_area < 0.001:
                    row_cells_data.append("Ngoài RG")
                    continue
                
                grid_lines_to_draw = []
                if isinstance(intersected_geo, Polygon):
                    grid_lines_to_draw.append(list(intersected_geo.exterior.coords))
                elif isinstance(intersected_geo, MultiPolygon):
                    for poly in intersected_geo.geoms:
                        grid_lines_to_draw.append(list(poly.exterior.coords))
                
                # --- THUẬT TOÁN MỚI: TÍCH PHÂN SỐ TRÊN LƯỚI SIÊU MỊN CỦA Ô LƯỚI ---
                sub_x = np.arange(x_start + SUB_STEP/2, x_end, SUB_STEP)
                sub_y = np.arange(y_start + SUB_STEP/2, y_end, SUB_STEP)
                xv, yv = np.meshgrid(sub_x, sub_y)
                sub_pts = np.vstack([xv.ravel(), yv.ravel()]).T
                
                # Lọc các điểm vi phân nằm thực tế bên trong ranh giới
                valid_sub_mask = np.array([boundary_polygon.contains(Point(p[0], p[1])) for p in sub_pts])
                if not np.any(valid_sub_mask):
                    row_cells_data.append("Ngoài RG")
                    continue
                    
                valid_sub_pts = sub_pts[valid_sub_mask]
                sub_area = SUB_STEP * SUB_STEP # Diện tích hình học của 1 pixel vi phân
                
                # Nội suy đồng bộ cao độ cho tập điểm vi phân
                def get_sub_z(pts_data, surface_cfg):
                    if surface_cfg["type"] == "const":
                        return np.full(len(valid_sub_pts), float(surface_cfg["value"]))
                    z = griddata(pts_data[:, :2], pts_data[:, 2], valid_sub_pts, method='linear')
                    nan_m = np.isnan(z)
                    if np.any(nan_m):
                        z[nan_m] = griddata(pts_data[:, :2], pts_data[:, 2], valid_sub_pts[nan_m], method='nearest')
                    return z.astype(float)
                    
                z1_sub = get_sub_z(pts1, surface_1)
                z2_sub = get_sub_z(pts2, surface_2)
                
                # Tích phân thể tích của ô lưới = Tổng thể tích các cột vi phân
                dz_sub = z2_sub - z1_sub
                cell_volume = np.sum(dz_sub * sub_area)
                
                cell_cut = abs(np.sum(dz_sub[dz_sub < 0] * sub_area))
                cell_fill = np.sum(dz_sub[dz_sub > 0] * sub_area)
                
                if cell_volume < 0:
                    cell_str = f"Đào: {cell_cut:.1f} m³"
                else:
                    cell_str = f"Đắp: {cell_fill:.1f} m³"
                    
                total_cut_vol += cell_cut
                total_fill_vol += cell_fill
                row_cells_data.append(f"{cell_str} ({actual_area:.1f}㎡)")
                
                cx, cy = intersected_geo.centroid.x, intersected_geo.centroid.y
                cad_cells.append({
                    'lines': grid_lines_to_draw, 'cx': cx, 'cy': cy,
                    'volume': -cell_cut if cell_volume < 0 else cell_fill
                })
                
            grid_rows_list.append(row_cells_data)
            
        if len(grid_rows_list) > 0:
            max_cols = max(len(r) for r in grid_rows_list)
            df_cols = [f"Cột {c+1}" for c in range(max_cols)]
            df_index = [f"Hàng {r+1}" for r in range(len(grid_rows_list))]
            st.session_state.df_result = pd.DataFrame(grid_rows_list, columns=df_cols, index=df_index).reset_index().rename(columns={'index': 'Hàng/Cột'})
            st.session_state.total_cut = total_cut_vol
            st.session_state.total_fill = total_fill_vol
            st.session_state.cad_grid_data = cad_cells
            st.session_state.calculated = True
if st.session_state.calculated and st.session_state.df_result is not None:
    st.success("🎉 Đã hoàn thành thuật toán tích phân vi phân lưới bảo toàn khối lượng tổng thể!")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Tổng khối lượng ĐÀO 🟥", f"{st.session_state.total_cut:,.2f} m³")
    col2.metric("Tổng khối lượng ĐẮP 🟩", f"{st.session_state.total_fill:,.2f} m³")
    net_diff = st.session_state.total_fill - st.session_state.total_cut
    col3.metric("Khối lượng cân bằng chênh lệch", f"{net_diff:,.2f} m³", delta_color="inverse")

    st.subheader("📊 Bảng phân bố lưới ô vuông đã cắt tỉa")
    st.dataframe(st.session_state.df_result, use_container_width=True)
    
    st.subheader("💾 Tải về file thành phẩm tích hợp số liệu thực")
    dwn_col1, dwn_col2 = st.columns(2)
    
    output_excel = io.BytesIO()
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        st.session_state.df_result.to_excel(writer, index=False, sheet_name="Khoi_Luong_Cat_Tia")
    excel_data = output_excel.getvalue()
    
    with dwn_col1:
        st.download_button(
            label="📥 Tải xuống Bảng tính Excel (.xlsx)",
            data=excel_data,
            file_name="khoi_luong_luoi_o_vuong.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
        
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()

    doc.layers.new(name='SURFACE_1', dxfattribs={'color': 1})    
    doc.layers.new(name='SURFACE_2', dxfattribs={'color': 3})    
    doc.layers.new(name='GRID_LINES', dxfattribs={'color': 7})   
    doc.layers.new(name='BOUNDARY_CUSTOM', dxfattribs={'color': 2}) 
    doc.layers.new(name='EARTHWORK_CUT', dxfattribs={'color': 1}) 
    doc.layers.new(name='EARTHWORK_FILL', dxfattribs={'color': 3})

    if st.session_state.pts1_real is not None:
        for pt in st.session_state.pts1_real:
            x, y, z = float(pt[0]), float(pt[1]), float(pt[2])
            msp.add_point((x, y, z), dxfattribs={'layer': 'SURFACE_1'})
            msp.add_text(text=f"{z:.2f}", dxfattribs={'layer': 'SURFACE_1', 'height': 0.3}).set_placement((x + 0.2, y, z))

    if st.session_state.pts2_real is not None:
        for pt in st.session_state.pts2_real:
            x, y, z = float(pt[0]), float(pt[1]), float(pt[2])
            msp.add_point((x, y, z), dxfattribs={'layer': 'SURFACE_2'})
            msp.add_text(text=f"{z:.2f}", dxfattribs={'layer': 'SURFACE_2', 'height': 0.3}).set_placement((x + 0.2, y, z))

    if st.session_state.boundary_poly_coords is not None:
        if st.session_state.boundary_source in ["surface2", "surfaces_intersect"]:
            msp.add_lwpolyline(st.session_state.boundary_poly_coords, dxfattribs={'layer': 'SURFACE_2', 'color': 2, 'const_width': 0.15})
        else:
            msp.add_lwpolyline(st.session_state.boundary_poly_coords, dxfattribs={'layer': 'BOUNDARY_CUSTOM', 'const_width': 0.15})

    for cell in st.session_state.cad_grid_data:
        for poly_line in cell['lines']:
            for i in range(len(poly_line) - 1):
                msp.add_line(poly_line[i], poly_line[i+1], dxfattribs={'layer': 'GRID_LINES'})
                
        cx, cy = cell['cx'], cell['cy']
        val = cell['volume']
        
        if val < 0:
            text_str = f"Dao: {abs(val):.1f}m3"
            t_obj = msp.add_text(text=text_str, dxfattribs={'layer': 'EARTHWORK_CUT', 'height': 0.3})
            t_obj.set_placement((cx, cy), align=TextEntityAlignment.MIDDLE_CENTER)
        else:
            text_str = f"Dap: {val:.1f}m3"
            t_obj = msp.add_text(text=text_str, dxfattribs={'layer': 'EARTHWORK_FILL', 'height': 0.3})
            t_obj.set_placement((cx, cy), align=TextEntityAlignment.MIDDLE_CENTER)

    output_dxf = io.StringIO()
    doc.write(output_dxf)
    dxf_data = output_dxf.getvalue().encode('utf-8')
    
    with dwn_col2:
        st.download_button(
            label="📥 Tải xuống Bản vẽ CAD Lưới Ô Vuông (.dxf)",
            data=dxf_data,
            file_name="khoi_luong_hoan_thien.dxf",
            mime="application/dxf",
            use_container_width=True
        )
else:
    st.info("💡 Hướng dẫn: Cấu hình các thông số bề mặt ở thanh điều hướng bên trái (Sidebar), sau đó nhấn nút 'Tiến hành tính toán khối lượng' để xem kết quả lưới ô vuông.")
