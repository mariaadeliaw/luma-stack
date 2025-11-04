import streamlit as st
import pandas as pd
import geopandas as gpd
import ee
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
import json
import tempfile
import zipfile
import os
import math
from shapely.geometry import shape, Point, Polygon, mapping
from epistemx.module_3 import InputCheck, SyncTrainData, SplitTrainData, LULCSamplingTool
from epistemx.ee_config import initialize_earth_engine

initialize_earth_engine()

# Initialize session state variables
session_defaults = {
    'sampling_data': {'type': 'FeatureCollection', 'features': []},
    'feature_count': 0,
    'last_recorded_feature': None,
    'center_lat': 0,
    'center_lon': 0,
    'map_zoom': 2,
    'initial_fit_done': False,
    'training_gdf': None,
    'active_tab': 0,
    'show_aoi_layer': False,
    'show_geotiff_layer': False
}

for key, default_value in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

st.title("Penentuan Data Sampel Klasifikasi Tutupan/penggunaan lahan")
st.divider()
st.markdown("Modul ini memungkinkan Anda untuk menyiapkan dan menentukan data sampel yang digunakan untuk proses klasifikasi tutupan/penggunaan lahan. "
    "Untuk menggunakan modul ini, hasil dari modul 1 dan 2 harus sudah tersedia. Jika sudah terpenuhi, Anda dapat:")
st.markdown("1. Mengunggah data sampel pelatihan.\n2. Membuat data sampel pelatihan melalui sampling on screen.\n3. Menggunakan data sampel default Epistem.")

st.sidebar.title("Tentang")
st.sidebar.info("Modul ini dibuat untuk menentukan data sampel pelatihan.")
st.sidebar.image("logos/logo_epistem.png")

st.markdown("Ketersediaan keluaran hasil modul 1 dan 2")
col1, col2, col3 = st.columns(3)

aoi_available = 'AOI' in st.session_state and 'gdf' in st.session_state
classification_available = 'classes' in st.session_state and len(st.session_state['classes']) > 0
composite_available = 'composite' in st.session_state

with col1:
    if aoi_available:
        st.success("✅ Data AOI dari modul 1 tersedia")
    else:
        st.error("❌ Data AOI belum tersedia, silakan kunjungi modul 1.")

with col2:
    if classification_available:
        class_count = len(st.session_state['classes'])
        scheme_type = "Skema Default" if st.session_state.get('ReferenceDataSource', False) else "Skema Kustom"
        st.success(f"✅ Data skema klasifikasi dari modul 2 tersedia ({scheme_type}) - {class_count} kelas")
    else:
        st.error("❌ Data skema klasifikasi belum tersedia, silakan kunjungi modul 2.")

with col3:
    if composite_available:
        st.success("✅ Data komposit citra dari modul 1 tersedia")
    else:
        st.error("❌ Data komposit citra belum tersedia, silakan kunjungi modul 1.")

if not (aoi_available and classification_available and composite_available):
    st.stop()

AOI = st.session_state.get('AOI')
AOI_GDF = st.session_state.get('gdf')
st.session_state['geotiff_overlay'] = st.session_state['composite']

LULCTable = pd.DataFrame(st.session_state['classes']) if 'classes' in st.session_state and len(st.session_state['classes']) > 0 else pd.DataFrame()
if LULCTable.empty:
    st.error("❌ Data klasifikasi tidak dapat dimuat dengan benar. Silakan kembali ke Modul 2 dan pastikan skema klasifikasi telah disimpan.")
    st.stop()

st.session_state['LULCTable'] = LULCTable
reference_data_source = st.session_state.get('ReferenceDataSource', False)



st.divider()
TrainField = 'LULC_Type'

if reference_data_source:
    st.info("🔄 Berdasarkan pilihan skema klasifikasi default di Modul 2, sistem akan menggunakan data sampel default Epistem.")
    st.subheader("A. Gunakan data sampel default (Epistem)")
    st.markdown("Data pelatihan akan dimuat dari dataset referensi RESTORE+ yang sesuai dengan skema klasifikasi yang dipilih.")
    
    TrainEePath = 'projects/ee-rg2icraf/assets/Indonesia_lulc_Sample'
    TrainField = 'kelas'
    
    if st.button("Muat Data Pelatihan Referensi", type="primary"):
        try:
            with st.spinner("Memuat dan memproses data pelatihan referensi..."):
                if AOI_GDF is not None:
                    if AOI_GDF.crs != 'EPSG:4326':
                        AOI_GDF_wgs84 = AOI_GDF.to_crs('EPSG:4326')
                        
                        from shapely.geometry import mapping
                        geom_list = []
                        for idx, row in AOI_GDF_wgs84.iterrows():
                            geom_dict = mapping(row.geometry)
                            geom_list.append(geom_dict)
                        
                        if len(geom_list) == 1:
                            AOI = ee.Geometry(geom_list[0])
                        else:
                            AOI = ee.Geometry.MultiPolygon(geom_list)
                        bounds = AOI_GDF_wgs84.total_bounds
                    else:
                        AOI_GDF_wgs84 = AOI_GDF
                        bounds = AOI_GDF.total_bounds
                    
                    if (bounds[0] < 90 or bounds[0] > 145 or
                        bounds[2] < 90 or bounds[2] > 145 or
                        bounds[1] < -15 or bounds[1] > 10 or
                        bounds[3] < -15 or bounds[3] > 10):
                        st.warning("⚠️ AOI bounds appear to be outside Indonesia region")
                        st.warning("Hal ini dapat menyebabkan masalah dalam memuat data pelatihan")
                    
                    area_deg2 = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
                    if area_deg2 > 10:
                        st.warning(f"⚠️ AOI covers a very large area ({area_deg2:.2f} deg²)")
                        st.warning("This might cause performance issues or timeouts")
                    
                    invalid_geoms = AOI_GDF_wgs84[~AOI_GDF_wgs84.geometry.is_valid]
                    if len(invalid_geoms) > 0:
                        st.warning(f"⚠️ Found {len(invalid_geoms)} invalid geometries - attempting to fix...")
                        AOI_GDF_wgs84.geometry = AOI_GDF_wgs84.geometry.buffer(0)
                
                TrainDataDict = {
                    'training_data': None,
                    'landcover_df': LULCTable,
                    'class_field': TrainField,
                    'validation_results': {
                        'total_points': 0,
                        'valid_points': 0,
                        'points_after_class_filter': 0,
                        'invalid_classes': [],
                        'outside_aoi': [],
                        'insufficient_samples': [],
                        'warnings': []
                    }
                }
                

                
                try:
                    TrainDataDict = SyncTrainData.LoadTrainData(
                        landcover_df=LULCTable,
                        aoi_geometry=AOI,
                        training_shp_path=None,
                        training_ee_path=TrainEePath
                    )
                except Exception as load_error:
                    st.error(f"❌ LoadTrainData failed: {str(load_error)}")
                    TrainDataDict = {
                        'training_data': gpd.GeoDataFrame(columns=['kelas', 'geometry']),
                        'landcover_df': LULCTable,
                        'class_field': 'kelas',
                        'validation_results': {
                            'total_points': 0,
                            'valid_points': 0,
                            'points_after_class_filter': 0,
                            'invalid_classes': [],
                            'outside_aoi': [],
                            'insufficient_samples': [],
                            'warnings': [f'LoadTrainData error: {str(load_error)}']
                        }
                    }
                
                if TrainDataDict and TrainDataDict.get('training_data') is not None:
                    training_data = TrainDataDict['training_data']
                    if hasattr(training_data, '__len__'):
                        train_count = len(training_data)
                    elif hasattr(training_data, 'size'):
                        train_count = training_data.size().getInfo() if hasattr(training_data.size(), 'getInfo') else 0
                    else:
                        train_count = 0
                    
                    if train_count > 0:
                        st.success(f"✅ Berhasil memuat {train_count} sampel pelatihan")
                    else:
                        st.warning("⚠️ Tidak ditemukan sampel pelatihan untuk AOI ini")
                        st.info("Hal ini dapat terjadi karena:")
                        st.info("• AOI Anda tidak tumpang tindih dengan area cakupan data pelatihan")
                        st.info("• Data pelatihan tidak memiliki sampel di wilayah spesifik Anda")
                        st.info("• Geometri AOI bermasalah")
                        TrainDataDict = {
                            'training_data': gpd.GeoDataFrame(columns=['kelas', 'geometry']),
                            'landcover_df': TrainDataDict.get('landcover_df', LULCTable),
                            'class_field': 'kelas',
                            'validation_results': {
                                'total_points': 0,
                                'valid_points': 0,
                                'points_after_class_filter': 0,
                                'invalid_classes': [],
                                'outside_aoi': [],
                                'insufficient_samples': [],
                                'warnings': ['No training data found for this AOI']
                            }
                        }
                        train_count = 0
                else:
                    st.warning("⚠️ Tidak ada data pelatihan yang dikembalikan dari Earth Engine")
                    st.info("Hal ini dapat disebabkan oleh:")
                    st.info("• Masalah autentikasi Earth Engine")
                    st.info("• Masalah konektivitas jaringan")
                    st.info("• Izin akses aset")
                    TrainDataDict = {
                        'training_data': gpd.GeoDataFrame(columns=['kelas', 'geometry']),
                        'landcover_df': LULCTable,
                        'class_field': 'kelas',
                        'validation_results': {
                            'total_points': 0,
                            'valid_points': 0,
                            'points_after_class_filter': 0,
                            'invalid_classes': [],
                            'outside_aoi': [],
                            'insufficient_samples': [],
                            'warnings': ['Earth Engine data loading failed']
                        }
                    }
                    train_count = 0
                
                if train_count == 0:
                    st.info("💡 **Saran untuk mendapatkan data pelatihan:**")
                    st.info("1. **Coba AOI yang berbeda** - Gunakan AOI yang mencakup area dengan tipe tutupan lahan yang diketahui")
                    st.info("2. **Gunakan Upload Data Sampel** - Unggah shapefile data pelatihan Anda sendiri")
                    st.info("3. **Gunakan Sampling On Screen** - Buat sampel pelatihan secara manual di peta")
                    st.info("4. **Periksa lokasi AOI** - Pastikan AOI Anda berada di dalam Indonesia dan memiliki geometri yang valid")
                
                if AOI_GDF is not None:
                    TrainDataDict['aoi_geometry'] = AOI_GDF
                

                
                progress = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Langkah 1/5: Mengatur field kelas...")
                if TrainDataDict.get('training_data') is not None:
                    TrainDataDict = SyncTrainData.SetClassField(TrainDataDict, TrainField)
                progress.progress(20)
                
                status_text.text("Langkah 2/5: Memvalidasi kelas...")
                if TrainDataDict.get('training_data') is not None:
                    TrainDataDict = SyncTrainData.ValidClass(TrainDataDict, use_class_ids=True)
                progress.progress(40)
                
                status_text.text("Langkah 3/5: Memeriksa kecukupan sampel...")
                if TrainDataDict.get('training_data') is not None:
                    TrainDataDict = SyncTrainData.CheckSufficiency(TrainDataDict, min_samples=20)
                progress.progress(60)
                
                # Filter by AOI
                status_text.text("Langkah 4/5: Memfilter berdasarkan AOI...")
                if TrainDataDict.get('training_data') is not None and AOI_GDF is not None:
                    TrainDataDict['aoi_geometry'] = AOI_GDF
                    TrainDataDict = SyncTrainData.FilterTrainAoi(TrainDataDict)
                progress.progress(80)
                
                status_text.text("Langkah 5/5: Membuat ringkasan...")
                if TrainDataDict.get('training_data') is not None:
                    table_df, total_samples, insufficient_df = SyncTrainData.TrainDataRaw(
                        training_data=TrainDataDict.get('training_data'),
                        landcover_df=TrainDataDict.get('landcover_df'),
                        class_field=TrainDataDict.get('class_field')
                    )
                    
                    st.session_state['table_df_ref'] = table_df
                    st.session_state['total_samples_ref'] = total_samples
                    st.session_state['insufficient_df_ref'] = insufficient_df
                    st.session_state['train_data_final_ref'] = TrainDataDict.get('training_data')
                    st.session_state['train_data_dict_ref'] = TrainDataDict
                
                progress.progress(100)
                status_text.text("Pemrosesan selesai!")
                st.session_state['data_processed_ref'] = True
                st.session_state['reference_data_loaded'] = True
                st.success("Data pelatihan referensi berhasil dimuat dan diproses!")
                st.rerun()
                
        except Exception as e:
            st.error(f"Error memuat data pelatihan referensi: {e}")
    
    if st.session_state.get('data_processed_ref', False):
            st.divider()
            st.subheader("B. Ringkasan Data Pelatihan")
            
            TrainDataDict = st.session_state.get('train_data_dict_ref', {})
            vr = TrainDataDict.get('validation_results', {})
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Titik Dimuat", vr.get('total_points', 'N/A'))
            with col2:
                st.metric("Titik Setelah Filter Kelas", vr.get('points_after_class_filter', 'N/A'))
            with col3:
                st.metric("Titik Valid (dalam AOI)", vr.get('valid_points', 'N/A'))
            with col4:
                st.metric("Kelas Invalid", len(vr.get('invalid_classes', [])))
            
            if 'table_df_ref' in st.session_state and st.session_state['table_df_ref'] is not None:
                st.markdown("#### Distribusi Data Pelatihan")
                
                # Format percentage column
                display_df = st.session_state['table_df_ref'].copy()
                if 'Percentage' in display_df.columns:
                    display_df['Percentage'] = display_df['Percentage'].apply(
                        lambda x: f"{x:.2f}%" if isinstance(x, (int, float)) else x
                    )
                
                st.dataframe(display_df, use_container_width=True)
                
                if 'insufficient_df_ref' in st.session_state and st.session_state['insufficient_df_ref'] is not None:
                    if len(st.session_state['insufficient_df_ref']) > 0:
                        st.warning(f"⚠️ {len(st.session_state['insufficient_df_ref'])} kelas memiliki sampel yang tidak mencukupi (< 20 sampel)")
                        
                        with st.expander("Lihat Kelas yang Tidak Mencukupi"):
                            insufficient_display = st.session_state['insufficient_df_ref'].copy()
                            if 'Percentage' in insufficient_display.columns:
                                insufficient_display['Percentage'] = insufficient_display['Percentage'].apply(
                                    lambda x: f"{x:.2f}%" if isinstance(x, (int, float)) else x
                                )
                            st.dataframe(insufficient_display, use_container_width=True)
            
            st.divider()
            st.subheader("C. Pembagian Data Pelatihan/Validasi")
            
            split_data = st.checkbox("Bagi data menjadi set pelatihan dan validasi", value=True, key="split_ref")
            
            if split_data:
                split_ratio = st.slider("Persentase data pelatihan:", min_value=0.5, max_value=0.9, value=0.7, step=0.05, key="split_ratio_ref")
                
                if st.button("Bagi Data Pelatihan", type="primary", key="split_ref_data"):
                    try:
                        if 'train_data_final_ref' in st.session_state and st.session_state['train_data_final_ref'] is not None:
                            train_data = st.session_state['train_data_final_ref']
                            
                            # Perform split
                            train_final, valid_final = SplitTrainData.SplitProcess(
                                train_data,
                                TrainSplitPct=split_ratio,
                                random_state=123
                            )
                            
                            st.session_state['train_final_ref'] = train_final
                            st.session_state['valid_final_ref'] = valid_final
                            st.session_state['split_completed_ref'] = True
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Sampel Pelatihan", len(train_final))
                            with col2:
                                st.metric("Sampel Validasi", len(valid_final))
                            with col3:
                                st.metric("Total Sampel", len(train_data))
                            
                            st.success("Pembagian data berhasil diselesaikan!")
                            
                            with st.expander("Preview Data yang Dibagi"):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown("**Preview Data Pelatihan:**")
                                    st.dataframe(train_final.head())
                                with col2:
                                    st.markdown("**Preview Data Validasi:**")
                                    st.dataframe(valid_final.head())
                                    
                    except Exception as e:
                        st.error(f"Error membagi data: {e}")
            else:
                if 'train_data_final_ref' in st.session_state and st.session_state['train_data_final_ref'] is not None:
                    st.session_state['train_final_ref'] = st.session_state['train_data_final_ref']
                    st.session_state['valid_final_ref'] = None
                    st.session_state['split_completed_ref'] = True
                    st.info("Menggunakan seluruh dataset sebagai data pelatihan.")

else:
    st.subheader("Pilih Mode Pengumpulan Data Pelatihan")
    if st.session_state.get('switch_to_tab2', False):
        st.session_state['switch_to_tab2'] = False
        st.session_state['active_tab'] = 1
    
    active_tab = st.session_state.get('active_tab', 0)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📤 Unggah Data Sampel", type="primary" if active_tab == 0 else "secondary", use_container_width=True, key="tab_button_1"):
            st.session_state['active_tab'] = 0
            st.rerun()
    with col2:
        if st.button("🎯 Sampling On Screen", type="primary" if active_tab == 1 else "secondary", use_container_width=True, key="tab_button_2"):
            st.session_state['active_tab'] = 1
            st.rerun()
    
    st.info(f"📍 **Saat ini di:** {'Unggah Data Sampel' if active_tab == 0 else 'Sampling On Screen'}")
    st.divider()
    
    # Show content based on active tab
    if active_tab == 0:
        st.subheader("A. Unggah data sampel (Shapefile)")
        st.markdown("Silakan unggah data shapefile terkompresi dalam format .zip")
        uploaded_file = st.file_uploader("Unggah shapefile (.zip)", type=["zip"])
        
        if uploaded_file:
            import tempfile
            import zipfile
            import os
            
            with tempfile.TemporaryDirectory() as tmpdir:
                # Extract zip file
                zip_path = os.path.join(tmpdir, "training.zip")
                with open(zip_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(tmpdir)
                
                # Find shapefile
                shp_files = []
                for root, _, files in os.walk(tmpdir):
                    for fname in files:
                        if fname.lower().endswith(".shp"):
                            shp_files.append(os.path.join(root, fname))
                
                if shp_files:
                    try:
                        # Load and validate shapefile
                        import geopandas as gpd
                        gdf = gpd.read_file(shp_files[0])
                        st.success("Data pelatihan berhasil dimuat!")

                        TrainField = st.selectbox(
                            "Pilih field yang berisi informasi nama kelas:",
                            options=["-- Pilih Field --"] + gdf.columns.tolist(),
                            help="Field ini harus berisi informasi nama kelas yang sesuai dengan skema klasifikasi Anda"
                        )
                        
                        if TrainField == "-- Pilih Field --":
                            TrainField = None
                                                                                              
                        if TrainField and TrainField != "-- Pilih Field --":
                            st.markdown("**Preview data pelatihan (tabel):**")
                            st.dataframe(gdf)

                            st.markdown("**Preview data pelatihan (peta):**")
                            import folium
                            from streamlit_folium import st_folium
                            
                            # Initialize map
                            m = folium.Map(tiles="OpenStreetMap")
                        
                            # Add basemap from module 1 if available
                            if st.session_state.geotiff_overlay is not None:
                                vis_params = {
                                    'bands': ['RED', 'GREEN', 'BLUE'],
                                    'min': 0,
                                    'max': 0.3
                                }
                                ee_image = st.session_state.geotiff_overlay.clip(AOI)
                                def add_ee_layer(self, ee_image_object, vis_params, name, opacity=1):
                                    map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
                                    folium.raster_layers.TileLayer(
                                        tiles=map_id_dict['tile_fetcher'].url_format,
                                        attr='Google Earth Engine',
                                        name=name,
                                        overlay=True,
                                        control=True,
                                        opacity=opacity
                                    ).add_to(self)
                                folium.Map.add_ee_layer = add_ee_layer
                                m.add_ee_layer(ee_image, vis_params, "Komposit Citra")
                        
                            # Add AOI layer
                            if AOI_GDF is not None:
                                aoi_gdf_wgs84 = AOI_GDF.to_crs('EPSG:4326') if AOI_GDF.crs != 'EPSG:4326' else AOI_GDF
                                folium.GeoJson(
                                    aoi_gdf_wgs84,
                                    name="AOI",
                                    style_function=lambda x: {'fillColor': 'transparent', 'color': '#FFD700', 'weight': 3}
                                ).add_to(m)
                        
                            # Add training data layer with colors
                            gdf_wgs84 = gdf.to_crs('EPSG:4326') if gdf.crs != 'EPSG:4326' else gdf
                            class_to_color = dict(zip(LULCTable['LULC_Type'], LULCTable['color_palette']))
                            class_to_id = dict(zip(LULCTable['LULC_Type'], LULCTable['ID']))
                            for idx, row in gdf_wgs84.iterrows():
                                class_value = row[TrainField] if TrainField in row else None
                                if isinstance(class_value, int):
                                    # Look up class name by ID
                                    matching_rows = LULCTable[LULCTable['ID'] == class_value]
                                    class_name = matching_rows['LULC_Type'].values[0] if len(matching_rows) > 0 else 'Unknown'
                                else:
                                    # Use the class value directly if it's a string and exists in color mapping
                                    class_name = class_value if class_value in class_to_color else 'Unknown'
                                color = class_to_color.get(class_name, '#808080')
                                geom = row.geometry
                                if geom.geom_type == 'Point':
                                    folium.Marker(
                                        location=[geom.y, geom.x],
                                        icon=folium.Icon(color='white', icon_color=color, icon='circle', prefix='fa'),
                                        popup=f"Class: {class_name}"
                                    ).add_to(m)
                                elif geom.geom_type == 'Polygon':
                                    folium.GeoJson(
                                        geom,
                                        name=class_name,
                                        style_function=lambda x, color=color: {'fillColor': color, 'color': color, 'weight': 2}
                                    ).add_to(m)
                        
                            # Add legend
                            legend_html = '''
                            <div style="
                                position: fixed; 
                                bottom: 10px; 
                                left: 10px; 
                                width: auto; 
                                height: auto; 
                                border: 1px solid #999; 
                                z-index: 9999; 
                                font-size: 12px; 
                                background-color: rgba(255, 255, 255, 0.9); 
                                padding: 10px 14px; 
                                border-radius: 6px; 
                                box-shadow: 0 2px 6px rgba(0,0,0,0.3);
                            ">
                                <b style="font-size: 13px; display: block; margin-bottom: 8px;">Legend</b>
                            '''

                            for class_name, color in class_to_color.items():
                                legend_html += f'''
                                <div style="display: flex; align-items: center; margin-bottom: 4px;">
                                    <div style="
                                        background: {color}; 
                                        width: 16px; 
                                        height: 16px; 
                                        margin-right: 8px; 
                                        border: 1px solid #444; 
                                        border-radius: 3px;
                                    "></div>
                                    <span style="font-size: 12px; white-space: nowrap;">{class_name}</span>
                                </div>
                                '''

                            legend_html += '</div>'
                            m.get_root().html.add_child(folium.Element(legend_html))

                            # Fit bounds to data or AOI
                            if not gdf_wgs84.empty:
                                bounds = gdf_wgs84.total_bounds
                            elif AOI_GDF is not None:
                                bounds = aoi_gdf_wgs84.total_bounds
                            else:
                                bounds = None
                            if bounds is not None:
                                m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
                            
                            # Layer control
                            folium.LayerControl().add_to(m)
                            
                            # Display map
                            st_folium(m, width=None, height=400)
                            
                            if st.button("Proses Data Pelatihan", type="primary", key="process_uploaded_data"):
                                # Store the data first
                                st.session_state['training_gdf'] = gdf
                                st.session_state['training_class_field'] = TrainField
                                
                                # Create TrainDataDict for processing
                                TrainDataDict = {
                                    'training_data': gdf,
                                    'landcover_df': LULCTable,
                                    'class_field': TrainField,
                                    'validation_results': {
                                        'total_points': len(gdf) if gdf is not None else 0,
                                        'valid_points': 0,
                                        'points_after_class_filter': 0,
                                        'invalid_classes': [],
                                        'outside_aoi': [],
                                        'insufficient_samples': [],
                                        'warnings': []
                                    }
                                }
                                
                                try:
                                    with st.spinner("Memproses data pelatihan..."):
                                        progress = st.progress(0)
                                        status_text = st.empty()
                                        
                                        # Set class field
                                        status_text.text("Langkah 1/5: Mengatur field kelas...")
                                        if TrainDataDict.get('training_data') is not None:
                                            TrainDataDict = SyncTrainData.SetClassField(TrainDataDict, TrainField)
                                        progress.progress(20)
                                        
                                        # Validate classes
                                        status_text.text("Langkah 2/5: Memvalidasi kelas...")
                                        if TrainDataDict.get('training_data') is not None:
                                            TrainDataDict = SyncTrainData.ValidClass(TrainDataDict, use_class_ids=False)
                                        progress.progress(40)
                                        
                                        # Check sufficiency
                                        status_text.text("Langkah 3/5: Memeriksa kecukupan sampel...")
                                        if TrainDataDict.get('training_data') is not None:
                                            TrainDataDict = SyncTrainData.CheckSufficiency(TrainDataDict, min_samples=20)
                                        progress.progress(60)
                                        
                                        # Filter by AOI
                                        status_text.text("Langkah 4/5: Memfilter berdasarkan AOI...")
                                        if TrainDataDict.get('training_data') is not None and AOI_GDF is not None:
                                            TrainDataDict['aoi_geometry'] = AOI_GDF
                                            TrainDataDict = SyncTrainData.FilterTrainAoi(TrainDataDict)
                                        progress.progress(80)
                                        
                                        # Create summary table
                                        status_text.text("Langkah 5/5: Membuat ringkasan...")
                                        if TrainDataDict.get('training_data') is not None:
                                            table_df, total_samples, insufficient_df = SyncTrainData.TrainDataRaw(
                                                training_data=TrainDataDict.get('training_data'),
                                                landcover_df=TrainDataDict.get('landcover_df'),
                                                class_field=TrainDataDict.get('class_field')
                                            )
                                            
                                            st.session_state['table_df_upload'] = table_df
                                            st.session_state['total_samples_upload'] = total_samples
                                            st.session_state['insufficient_df_upload'] = insufficient_df
                                            st.session_state['train_data_final_upload'] = TrainDataDict.get('training_data')
                                            st.session_state['train_data_dict_upload'] = TrainDataDict
                                        
                                        progress.progress(100)
                                        status_text.text("Pemrosesan selesai!")
                                        st.session_state['data_processed_upload'] = True
                                        st.success("Data pelatihan berhasil diproses!")
                                        st.rerun()
                                        
                                except Exception as e:
                                    st.error(f"Error memproses data pelatihan: {e}")
                            
                    except Exception as e:
                        st.error(f"Error membaca shapefile: {e}")
                else:
                    st.error("File .shp tidak ditemukan dalam zip yang diunggah.")
        
        if st.session_state.get('data_processed_upload', False):
            st.divider()
            st.subheader("B. Ringkasan Data Pelatihan")
            
            TrainDataDict = st.session_state.get('train_data_dict_upload', {})
            vr = TrainDataDict.get('validation_results', {})
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Titik Dimuat", vr.get('total_points', 'N/A'))
            with col2:
                st.metric("Titik Setelah Filter Kelas", vr.get('points_after_class_filter', 'N/A'))
            with col3:
                st.metric("Titik Valid (dalam AOI)", vr.get('valid_points', 'N/A'))
            with col4:
                st.metric("Kelas Invalid", len(vr.get('invalid_classes', [])))
            
            if 'table_df_upload' in st.session_state and st.session_state['table_df_upload'] is not None:
                    st.markdown("#### Distribusi Data Pelatihan")
                    
                    # Format percentage column
                    display_df = st.session_state['table_df_upload'].copy()
                    if 'Percentage' in display_df.columns:
                        display_df['Percentage'] = display_df['Percentage'].apply(
                            lambda x: f"{x:.2f}%" if isinstance(x, (int, float)) else x
                        )
                    
                    st.dataframe(display_df, use_container_width=True)
                    
                    if 'insufficient_df_upload' in st.session_state and st.session_state['insufficient_df_upload'] is not None:
                        if len(st.session_state['insufficient_df_upload']) > 0:
                            st.warning(f"⚠️ {len(st.session_state['insufficient_df_upload'])} kelas memiliki sampel yang tidak mencukupi (< 20 sampel)")
                            
                            with st.expander("Lihat Kelas yang Tidak Mencukupi"):
                                insufficient_display = st.session_state['insufficient_df_upload'].copy()
                                if 'Percentage' in insufficient_display.columns:
                                    insufficient_display['Percentage'] = insufficient_display['Percentage'].apply(
                                        lambda x: f"{x:.2f}%" if isinstance(x, (int, float)) else x
                                    )
                                st.dataframe(insufficient_display, use_container_width=True)
                            
                            if st.button("🎯 Lengkapi data dengan sampling on screen", type="secondary", use_container_width=True):
                                if 'training_gdf' in st.session_state and st.session_state['training_gdf'] is not None:
                                    uploaded_gdf = st.session_state['training_gdf']
                                    class_field = st.session_state.get('training_class_field')
                                    
                                    if 'sampling_data' not in st.session_state:
                                        st.session_state.sampling_data = {'type': 'FeatureCollection', 'features': []}
                                    
                                    features_to_add = []
                                    feature_id_start = st.session_state.get('feature_count', 0) + 1
                                    
                                    for idx, row in uploaded_gdf.iterrows():
                                        class_value = row[class_field] if class_field in row else None
                                        if pd.isna(class_value):
                                            continue
                                            
                                        if isinstance(class_value, (int, float)):
                                            matching_rows = st.session_state['LULCTable'][st.session_state['LULCTable']['ID'] == class_value]
                                            if len(matching_rows) > 0:
                                                class_name = matching_rows['LULC_Type'].values[0]
                                                class_id = int(class_value)
                                                class_color = matching_rows['color_palette'].values[0]
                                            else:
                                                continue
                                        else:
                                            matching_rows = st.session_state['LULCTable'][st.session_state['LULCTable']['LULC_Type'] == class_value]
                                            if len(matching_rows) > 0:
                                                class_name = class_value
                                                class_id = int(matching_rows['ID'].values[0])
                                                class_color = matching_rows['color_palette'].values[0]
                                            else:
                                                continue
                                        
                                        geom = row.geometry
                                        if geom.geom_type == 'Polygon':
                                            geom = geom.centroid
                                        
                                        feature = {
                                            'type': 'Feature',
                                            'geometry': {
                                                'type': 'Point',
                                                'coordinates': [geom.x, geom.y]
                                            },
                                            'properties': {
                                                'feature_id': feature_id_start + len(features_to_add),
                                                'LULC_Class': class_name,
                                                'LULC_ID': class_id,
                                                'Class_Color': class_color,
                                                'source': 'uploaded'
                                            }
                                        }
                                        features_to_add.append(feature)
                                    
                                    st.session_state.sampling_data['features'].extend(features_to_add)
                                    st.session_state.feature_count = st.session_state.get('feature_count', 0) + len(features_to_add)
                                    st.session_state['switch_to_tab2'] = True
                                    st.session_state['uploaded_data_loaded_to_tab2'] = True
                                    
                                    st.success(f"✅ {len(features_to_add)} sampel dari data upload telah dimuat ke sampling on screen!")
                                    st.info("🔄 Beralih ke tab 'Sampling On Screen'...")
                                    st.rerun()
                
            st.divider()
            st.subheader("C. Pembagian Data Pelatihan/Validasi")
            
            split_data = st.checkbox("Bagi data menjadi set pelatihan dan validasi", value=True, key="split_upload")
            
            if split_data:
                split_ratio = st.slider("Persentase data pelatihan:", min_value=0.5, max_value=0.9, value=0.7, step=0.05, key="split_ratio_upload")
                
                if st.button("Bagi Data Pelatihan", type="primary", key="split_upload_data"):
                        try:
                            if 'train_data_final_upload' in st.session_state and st.session_state['train_data_final_upload'] is not None:
                                train_data = st.session_state['train_data_final_upload']
                                
                                # Perform split
                                train_final, valid_final = SplitTrainData.SplitProcess(
                                    train_data,
                                    TrainSplitPct=split_ratio,
                                    random_state=123
                                )
                                
                                st.session_state['train_final_upload'] = train_final
                                st.session_state['valid_final_upload'] = valid_final
                                st.session_state['split_completed_upload'] = True
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Sampel Pelatihan", len(train_final))
                                with col2:
                                    st.metric("Sampel Validasi", len(valid_final))
                                with col3:
                                    st.metric("Total Sampel", len(train_data))
                                
                                st.success("Pembagian data berhasil diselesaikan!")
                                
                                # Show preview of split data
                                with st.expander("Preview Data yang Dibagi"):
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.markdown("**Preview Data Training:**")
                                        st.dataframe(train_final.head())
                                    with col2:
                                        st.markdown("**Preview Data Validasi:**")
                                        st.dataframe(valid_final.head())
                                        
                        except Exception as e:
                            st.error(f"Error membagi data: {e}")
                else:
                    if 'train_data_final_upload' in st.session_state and st.session_state['train_data_final_upload'] is not None:
                        st.session_state['train_final_upload'] = st.session_state['train_data_final_upload']
                        st.session_state['valid_final_upload'] = None
                        st.session_state['split_completed_upload'] = True
                        st.info("Menggunakan seluruh dataset sebagai data pelatihan.")

    elif active_tab == 1:
        st.subheader("A. Buat data sampel (On Screen)")
        
        # Show notification if uploaded data has been loaded
        if st.session_state.get('uploaded_data_loaded_to_tab2', False):
            st.success("✅ Data dari upload telah dimuat! Anda dapat menambahkan sampel tambahan di bawah ini.")
            # Clear the flag after showing the message
            if st.button("✓ Mengerti", type="secondary"):
                st.session_state['uploaded_data_loaded_to_tab2'] = False
                st.rerun()
        
        st.info("Gunakan peta di bawah untuk mengumpulkan sampel pelatihan dengan menambahkan koordinat secara manual.")
        
        if 'classes_df' not in st.session_state:
            st.session_state.classes_df = dict(zip(LULCTable['LULC_Type'], LULCTable['color_palette']))
        classes_df = st.session_state.classes_df

        if st.session_state.classes_df is not None and LULCTable is not None:
            for feature in st.session_state.sampling_data['features']:
                if feature['properties'].get('source') == 'uploaded':
                    class_name = feature['properties']['LULC_Class']
                    matching_row = LULCTable[LULCTable['LULC_Type'] == class_name]
                    if not matching_row.empty:
                        feature['properties']['Class_Color'] = matching_row['color_palette'].iloc[0]
                        feature['properties']['LULC_ID'] = int(matching_row['ID'].iloc[0])
                    else:
                        feature['properties']['Class_Color'] = classes_df.get(class_name, '#808080')

        st.subheader("Map Layers")
        col1, col2 = st.columns(2)
        with col1:
            selected_class = st.selectbox("**Pilih Kelas LULC untuk Menggambar:**", options=list(classes_df.keys()))
        with col2:
            basemap_option = st.selectbox("**Peta Dasar:**", options=["Satellite (ESRI)", "CartoDB Dark", "Satellite (Google)", "OpenStreetMap", "CartoDB Positron"])
            
        def add_ee_layer(self, ee_image_object, vis_params, name, opacity=1.0):
            map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
            folium.raster_layers.TileLayer(tiles=map_id_dict['tile_fetcher'].url_format, attr='Google Earth Engine', 
                                         name=name, overlay=True, control=True, opacity=opacity).add_to(self)
        folium.Map.add_ee_layer = add_ee_layer

        st.session_state.show_aoi_layer = st.session_state.get('show_aoi_layer', AOI_GDF is not None)
        st.session_state.show_geotiff_layer = st.session_state.get('show_geotiff_layer', st.session_state.geotiff_overlay is not None)
        show_aoi, show_geotiff = st.session_state.show_aoi_layer, st.session_state.show_geotiff_layer

        map_center = [st.session_state.center_lat, st.session_state.center_lon]
        basemap_configs = {
            "Satellite (ESRI)": {"tiles": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", "attr": "ESRI"},
            "Satellite (Google)": {"tiles": "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}", "attr": "Google"},
            "OpenStreetMap": {"tiles": "OpenStreetMap"},
            "CartoDB Dark": {"tiles": "CartoDB dark_matter"},
            "CartoDB Positron": {"tiles": "CartoDB positron"}
        }
        config = basemap_configs.get(basemap_option, basemap_configs["CartoDB Positron"])
        m = folium.Map(location=map_center, zoom_start=st.session_state.map_zoom, **config)

        if AOI_GDF is not None:
            aoi_gdf_wgs84 = AOI_GDF.to_crs('EPSG:4326') if AOI_GDF.crs != 'EPSG:4326' else AOI_GDF
            bounds = aoi_gdf_wgs84.total_bounds
            st.session_state.center_lat, st.session_state.center_lon = (bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2
            max_diff = max(bounds[3] - bounds[1], bounds[2] - bounds[0])
            st.session_state.map_zoom = int(13 - math.log(max_diff + 0.01)) if max_diff > 0 else 5

        if show_geotiff and st.session_state.geotiff_overlay is not None:
            vis_params = {'bands': ['RED', 'GREEN', 'BLUE'], 'min': 0, 'max': 0.3}
            m.add_ee_layer(st.session_state.geotiff_overlay.clip(AOI), vis_params, "Custom Basemap", opacity=0.8)

        if show_aoi and AOI_GDF is not None:
            try:
                aoi_gdf_wgs84 = AOI_GDF.to_crs('EPSG:4326') if AOI_GDF.crs != 'EPSG:4326' else AOI_GDF
                folium.GeoJson(aoi_gdf_wgs84, name="AOI Layer", 
                             style_function=lambda x: {'fillColor': 'transparent', 'color': '#FFD700', 'weight': 3, 'dashArray': '5, 5', 'fillOpacity': 0.1},
                             tooltip="Area of Interest").add_to(m)
            except Exception as e:
                st.error(f"Error displaying AOI: {e}")

        feature_group = folium.FeatureGroup(name="LULC Features")
        for feature in st.session_state.sampling_data['features']:
            geom_type, color = feature['geometry']['type'], feature['properties'].get('Class_Color', '#808080')
            popup_text = f"Feature ID: {feature['properties'].get('feature_id', feature['properties'].get('id', 'N/A'))}<br>LULC ID: {feature['properties'].get('LULC_ID', 'N/A')}<br>LULC: {feature['properties'].get('LULC_Class', 'Unknown')}"
            
            if geom_type == 'Point':
                coords = feature['geometry']['coordinates']
                folium.Marker(location=[coords[1], coords[0]], icon=folium.Icon(color='white', icon_color=color, icon='circle', prefix='fa'), popup=popup_text).add_to(feature_group)
            elif geom_type == 'Polygon':
                latlngs = [[coord[1], coord[0]] for coord in feature['geometry']['coordinates'][0]]
                folium.Polygon(locations=latlngs, color=color, fill_color=color, fill_opacity=0.4, weight=3, popup=popup_text).add_to(feature_group)
        feature_group.add_to(m)

        draw_options = {'polyline': False, 'rectangle': False, 'circle': False, 'circlemarker': False, 'marker': True,
                       'polygon': {"allowIntersection": False, "shapeOptions": {"color": classes_df[selected_class], "fillColor": classes_df[selected_class], "fillOpacity": 0.5}}}
        Draw(export=False, position='topleft', draw_options=draw_options, edit_options={'edit': False, 'remove': False}).add_to(m)
        folium.LayerControl().add_to(m)

        col_map, col_colors = st.columns([3, 1])
        with col_map:
            map_output = st_folium(m, width=None, height=600, key="folium_map", returned_objects=["last_active_drawing", "center", "zoom"])
        
        with col_colors:
            st.markdown("**Visibilitas Layer:**")
            st.session_state.show_aoi_layer = st.checkbox("Tampilkan Layer AOI", value=st.session_state.show_aoi_layer, disabled=AOI_GDF is None)
            st.session_state.show_geotiff_layer = st.checkbox("Tampilkan Peta Dasar Kustom", value=st.session_state.show_geotiff_layer, disabled=st.session_state.geotiff_overlay is None)
            
            st.markdown("**Warna Kelas:**")
            if st.session_state.sampling_data['features']:
                try:
                    gdf_colors = gpd.GeoDataFrame.from_features(st.session_state.sampling_data['features'])
                    for class_name in classes_df:
                        total_count = len(gdf_colors[gdf_colors['LULC_Class'] == class_name])
                        if total_count > 0:
                            st.markdown(f"<span style='color: {classes_df[class_name]}; font-size: 16px;'>●</span> {class_name} ({total_count})", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error displaying class colors: {e}")
            else:
                for class_name, color in classes_df.items():
                    st.markdown(f"<span style='color: {color}; font-size: 16px;'>●</span> {class_name}", unsafe_allow_html=True)



        if map_output:
            if map_output.get("center"):
                st.session_state.center_lat, st.session_state.center_lon = map_output["center"]["lat"], map_output["center"]["lng"]
            if map_output.get("zoom"):
                st.session_state.map_zoom = map_output["zoom"]

        if map_output and map_output.get("last_active_drawing"):
            new_feature = map_output["last_active_drawing"]
            if new_feature['geometry'] != st.session_state.last_recorded_feature:
                st.session_state.feature_count += 1
                if 'properties' not in new_feature:
                    new_feature['properties'] = {}
                
                new_feature['properties'].update({
                    'feature_id': st.session_state.feature_count,
                    'LULC_Class': selected_class,
                    'Class_Color': classes_df[selected_class],
                    'source': 'digitized'
                })
                
                if st.session_state['LULCTable'] is not None:
                    try:
                        lulc_id = st.session_state['LULCTable'][st.session_state['LULCTable']['LULC_Type'] == selected_class]['ID'].iloc[0]
                        new_feature['properties']['LULC_ID'] = int(lulc_id)
                    except (IndexError, KeyError):
                        new_feature['properties']['LULC_ID'] = 0
                else:
                    new_feature['properties']['LULC_ID'] = 0
                
                st.session_state.sampling_data['features'].append(new_feature)
                st.session_state.last_recorded_feature = new_feature['geometry']
                st.success(f"✓ Captured Feature #{st.session_state.feature_count}: **{selected_class}** ({new_feature['geometry']['type']})")
                st.rerun()

        st.subheader("Fitur LULC yang Terekam")
        if st.session_state.sampling_data['features']:
            try:
                gdf = gpd.GeoDataFrame.from_features(st.session_state.sampling_data['features'])
                summary_data = []
                for class_name in classes_df:
                    class_features = gdf[gdf['LULC_Class'] == class_name]
                    uploaded_count = len(class_features[class_features.get('source', '') == 'uploaded'])
                    digitized_count = len(class_features[class_features.get('source', '') != 'uploaded'])
                    summary_data.append({'LULC Class': class_name, 'Uploaded': uploaded_count, 'Digitized': digitized_count, 'Total': len(class_features), 'Color': classes_df[class_name]})
                summary_df = pd.DataFrame(summary_data)
                        
                if 'feature_id' in gdf.columns:
                    gdf_display = gdf[['feature_id', 'LULC_ID', 'LULC_Class', 'Class_Color']].copy()
                else:
                    gdf_display = gdf[['id', 'LULC_Class']].copy()
                    gdf_display['feature_id'] = gdf_display['id']
                    gdf_display['LULC_ID'] = gdf_display.get('LULC_ID', 0)
                    gdf_display['Class_Color'] = gdf_display.get('Class_Color', '#808080')
                    gdf_display = gdf_display[['feature_id', 'LULC_ID', 'LULC_Class', 'Class_Color']].copy()
                
                gdf_display['Source'] = gdf['source'].fillna('digitized').apply(lambda x: 'Uploaded' if x == 'uploaded' else 'Digitized')
                gdf_display['Geometry_Type'] = gdf.geometry.apply(lambda geom: geom.geom_type)
                gdf_display['Geometry'] = gdf.geometry.apply(lambda geom: geom.wkt)
                gdf_display = gdf_display[['LULC_ID', 'LULC_Class', 'Class_Color', 'feature_id', 'Source', 'Geometry_Type', 'Geometry']]
                gdf_display = gdf_display.rename(columns={'LULC_ID': 'LULC ID', 'LULC_Class': 'LULC Class', 'Class_Color': 'Class Color', 'feature_id': 'Feature ID', 'Geometry_Type': 'Geometry Type'})
                
                col_left, col_right = st.columns([3, 1])
                with col_left:
                    st.markdown("**Tabel Fitur**")
                    st.dataframe(gdf_display, use_container_width=True, hide_index=True, height=300)
                
                with col_right:
                    st.markdown("**Hapus Fitur**")
                    feature_ids = gdf_display['Feature ID'].tolist()
                    selected_feature_id = st.selectbox("Pilih Feature ID:", feature_ids, label_visibility="collapsed")
                    if st.button("🗑️ Hapus", use_container_width=True, type="secondary"):
                        st.session_state.sampling_data['features'] = [f for f in st.session_state.sampling_data['features'] 
                                                                     if f['properties'].get('feature_id', f['properties'].get('id')) != selected_feature_id]
                        st.success(f"Feature ID #{selected_feature_id} telah dihapus")
                        st.rerun()
                    st.metric("Total Fitur", len(gdf_display))

                col2, col3, col4, col5, col6 = st.columns(5)
                with col2:
                    st.metric("Kelas LULC", gdf_display['LULC Class'].nunique())
                with col3:
                    st.metric("Titik", len(gdf_display[gdf_display['Geometry Type'] == 'Point']))
                with col4:
                    st.metric("Poligon", len(gdf_display[gdf_display['Geometry Type'] == 'Polygon']))
                with col5:
                    st.metric("Sampel Terunggah", summary_df['Uploaded'].sum())
                with col6:
                    st.metric("Sampel Digitasi", summary_df['Digitized'].sum())

                st.markdown("---")
                st.subheader("Unduh Data")
                col_a, col_b = st.columns(2)
                with col_a:
                    geojson_str = json.dumps(st.session_state.sampling_data, indent=2)
                    st.download_button("📥 Unduh GeoJSON", data=geojson_str, file_name="LULC_digitization_data.geojson", mime="application/json", use_container_width=True)
                
                with col_b:
                    if st.button("📥 Unduh Shapefile (.zip)", use_container_width=True):
                        gdf_export = gpd.GeoDataFrame.from_features(st.session_state.sampling_data['features'], crs='EPSG:4326')
                        with tempfile.TemporaryDirectory() as tmpdir:
                            shp_path = os.path.join(tmpdir, 'export.shp')
                            gdf_export.to_file(shp_path)
                            zip_path = os.path.join(tmpdir, 'export.zip')
                            with zipfile.ZipFile(zip_path, 'w') as zipf:
                                for ext in ['.shp', '.shx', '.dbf', '.prj']:
                                    file = shp_path.replace('.shp', ext)
                                    if os.path.exists(file):
                                        zipf.write(file, os.path.basename(file))
                            with open(zip_path, 'rb') as f:
                                zip_data = f.read()
                        st.download_button("Unduh Shapefile (.zip)", data=zip_data, file_name="lulc_sampling.zip", mime="application/zip", use_container_width=True)
                
            except Exception as e:
                st.error(f"Error processing GeoJSON data: {e}")
                st.json(st.session_state.sampling_data)
        else:
            st.info("Mulai menggambar titik atau poligon di peta untuk mengumpulkan data LULC. Tabel akan muncul di sini.")

        if st.session_state.sampling_data['features']:
            st.divider()
            st.subheader("B. Pemrosesan & Validasi Data Pelatihan")
            
            if st.button("Proses Sampel yang Dikumpulkan", type="primary", key="process_sampling_data"):
                try:
                    geometries = []
                    training_points = []
                    features = st.session_state.sampling_data['features']
                    for feature in features:
                        geom = shape(feature['geometry'])
                        if geom.geom_type == 'Polygon':
                            geom = geom.centroid
                        elif geom.geom_type != 'Point':
                            continue
                        training_points.append({
                            'kelas': feature['properties']['LULC_ID'],
                            'LULC_Type': feature['properties']['LULC_Class'],
                        })
                        geometries.append(geom)
                    
                    train_data_gdf = gpd.GeoDataFrame(training_points, geometry=geometries, crs='EPSG:4326')
                    
                    TrainField = 'kelas'
                    TrainDataDict = {
                        'training_data': train_data_gdf,
                        'landcover_df': LULCTable,
                        'class_field': TrainField,
                        'validation_results': {
                            'total_points': len(train_data_gdf),
                            'valid_points': 0,
                            'points_after_class_filter': 0,
                            'invalid_classes': [],
                            'outside_aoi': [],
                            'insufficient_samples': [],
                            'warnings': []
                        }
                    }
                    
                    progress = st.progress(0)
                    status_text = st.empty()
                    
                    # Set class field
                    status_text.text("Langkah 1/5: Mengatur field kelas...")
                    if TrainDataDict.get('training_data') is not None:
                        TrainDataDict = SyncTrainData.SetClassField(TrainDataDict, TrainField)
                    progress.progress(20)
                    
                    # Validate classes
                    status_text.text("Langkah 2/5: Memvalidasi kelas...")
                    if TrainDataDict.get('training_data') is not None:
                        TrainDataDict = SyncTrainData.ValidClass(TrainDataDict, use_class_ids=True)
                    progress.progress(40)
                    
                    # Check sufficiency
                    status_text.text("Langkah 3/5: Memeriksa kecukupan sampel...")
                    if TrainDataDict.get('training_data') is not None:
                        TrainDataDict = SyncTrainData.CheckSufficiency(TrainDataDict, min_samples=20)
                    progress.progress(60)
                    
                    # Filter by AOI
                    status_text.text("Langkah 4/5: Memfilter berdasarkan AOI...")
                    if TrainDataDict.get('training_data') is not None and AOI_GDF is not None:
                        TrainDataDict['aoi_geometry'] = AOI_GDF
                        TrainDataDict = SyncTrainData.FilterTrainAoi(TrainDataDict)
                    progress.progress(80)
                    
                    # Create summary table
                    status_text.text("Langkah 5/5: Membuat ringkasan...")
                    if TrainDataDict.get('training_data') is not None:
                        table_df, total_samples, insufficient_df = SyncTrainData.TrainDataRaw(
                            training_data=TrainDataDict.get('training_data'),
                            landcover_df=TrainDataDict.get('landcover_df'),
                            class_field=TrainDataDict.get('class_field')
                        )
                        
                        st.session_state['table_df_sampling'] = table_df
                        st.session_state['total_samples_sampling'] = total_samples
                        st.session_state['insufficient_df_sampling'] = insufficient_df
                        st.session_state['train_data_final_sampling'] = TrainDataDict.get('training_data')
                        st.session_state['train_data_dict_sampling'] = TrainDataDict
                    
                    progress.progress(100)
                    status_text.text("Pemrosesan selesai!")
                    st.session_state['data_processed_sampling'] = True
                    
                    st.success(f"Berhasil memproses {len(train_data_gdf)} titik pelatihan dari sampling!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error memproses sampel: {e}")
            
            if st.session_state.get('data_processed_sampling', False):
                st.divider()
                st.subheader("C. Ringkasan Data Pelatihan")
                
                TrainDataDict = st.session_state.get('train_data_dict_sampling', {})
                vr = TrainDataDict.get('validation_results', {})
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Titik Dimuat", vr.get('total_points', 'N/A'))
                with col2:
                    st.metric("Titik Setelah Filter Kelas", vr.get('points_after_class_filter', 'N/A'))
                with col3:
                    st.metric("Titik Valid (dalam AOI)", vr.get('valid_points', 'N/A'))
                with col4:
                    st.metric("Kelas Invalid", len(vr.get('invalid_classes', [])))
                
                if 'table_df_sampling' in st.session_state and st.session_state['table_df_sampling'] is not None:
                    st.markdown("#### Distribusi Data Pelatihan")
                    
                    # Format percentage column
                    display_df = st.session_state['table_df_sampling'].copy()
                    if 'Percentage' in display_df.columns:
                        display_df['Percentage'] = display_df['Percentage'].apply(
                            lambda x: f"{x:.2f}%" if isinstance(x, (int, float)) else x
                        )
                    
                    st.dataframe(display_df, use_container_width=True)
                    
                    if 'insufficient_df_sampling' in st.session_state and st.session_state['insufficient_df_sampling'] is not None:
                        if len(st.session_state['insufficient_df_sampling']) > 0:
                            st.warning(f"⚠️ {len(st.session_state['insufficient_df_sampling'])} kelas memiliki sampel yang tidak mencukupi (< 20 sampel)")
                            
                            with st.expander("Lihat Kelas yang Tidak Mencukupi"):
                                insufficient_display = st.session_state['insufficient_df_sampling'].copy()
                                if 'Percentage' in insufficient_display.columns:
                                    insufficient_display['Percentage'] = insufficient_display['Percentage'].apply(
                                        lambda x: f"{x:.2f}%" if isinstance(x, (int, float)) else x
                                    )
                                st.dataframe(insufficient_display, use_container_width=True)
                
                st.divider()
                st.subheader("D. Pembagian Data Pelatihan/Validasi")
                
                split_data = st.checkbox("Bagi data menjadi set pelatihan dan validasi", value=True, key="split_sampling")
                
                if split_data:
                    split_ratio = st.slider("Persentase data pelatihan:", min_value=0.5, max_value=0.9, value=0.7, step=0.05, key="split_ratio_sampling")
                    
                    if st.button("Bagi Data Pelatihan", type="primary", key="split_sampling_data"):
                        try:
                            if 'train_data_final_sampling' in st.session_state and st.session_state['train_data_final_sampling'] is not None:
                                train_data = st.session_state['train_data_final_sampling']
                                
                                # Perform split
                                train_final, valid_final = SplitTrainData.SplitProcess(
                                    train_data,
                                    TrainSplitPct=split_ratio,
                                    random_state=123
                                )
                                
                                st.session_state['train_final_sampling'] = train_final
                                st.session_state['valid_final_sampling'] = valid_final
                                st.session_state['split_completed_sampling'] = True
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Sampel Pelatihan", len(train_final))
                                with col2:
                                    st.metric("Sampel Validasi", len(valid_final))
                                with col3:
                                    st.metric("Total Sampel", len(train_data))
                                
                                st.success("Pembagian data berhasil diselesaikan!")
                                
                                # Show preview of split data
                                with st.expander("Preview Data yang Dibagi"):
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.markdown("**Preview Data Training:**")
                                        st.dataframe(train_final.head())
                                    with col2:
                                        st.markdown("**Preview Data Validasi:**")
                                        st.dataframe(valid_final.head())
                                        
                        except Exception as e:
                            st.error(f"Error membagi data: {e}")
                else:
                    if 'train_data_final_sampling' in st.session_state and st.session_state['train_data_final_sampling'] is not None:
                        st.session_state['train_final_sampling'] = st.session_state['train_data_final_sampling']
                        st.session_state['valid_final_sampling'] = None
                        st.session_state['split_completed_sampling'] = True
                        st.info("Menggunakan seluruh dataset sebagai data pelatihan.")

st.divider()
st.subheader("Navigasi Modul")
col1, col2 = st.columns(2)

with col1:
    if st.button("⬅️ Kembali ke Modul 2: Skema Klasifikasi", use_container_width=True):
        st.switch_page("pages/2_Module_2_Classification_scheme.py")

with col2:
    training_ready = any([st.session_state.get('data_processed_ref', False), st.session_state.get('data_processed_upload', False), st.session_state.get('data_processed_sampling', False)])
    if training_ready:
        if st.button("➡️ Lanjut ke Modul 4: Klasifikasi", type="primary", use_container_width=True):
            st.switch_page("pages/4_Module_4_Analyze_ROI.py")
    else:
        st.button("🔒 Selesaikan Data Pelatihan Dulu", disabled=True, use_container_width=True, help="Silakan proses data pelatihan sebelum melanjutkan")

if training_ready:
    data_sources = [('data_processed_ref', 'train_data_final_ref', 'Referensi'), ('data_processed_upload', 'train_data_final_upload', 'Upload'), ('data_processed_sampling', 'train_data_final_sampling', 'Sampling')]
    for processed_key, final_key, source_name in data_sources:
        if st.session_state.get(processed_key, False):
            total_samples = len(st.session_state.get(final_key, []))
            st.session_state.update({
                'train_data_dict': st.session_state.get(f'train_data_dict_{source_name.lower()}'),
                'train_final': st.session_state.get(f'train_final_{source_name.lower()}'),
                'valid_final': st.session_state.get(f'valid_final_{source_name.lower()}', None)
            })
            st.success(f"✅ Data pelatihan siap dengan {total_samples} sampel ({source_name})")
            break
else:
    st.info("Selesaikan pengumpulan dan pemrosesan data pelatihan untuk melanjutkan")