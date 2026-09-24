import re
import io

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
import pandas as pd, numpy as np, re, io, joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

BASE=Path(__file__).resolve().parent
DATA=BASE/"data"; MODELS=BASE/"models"; DATA.mkdir(exist_ok=True); MODELS.mkdir(exist_ok=True)
app=FastAPI(title="Hanoi House Price Demo")
app.mount("/static",StaticFiles(directory=BASE/"static"),name="static")
STATE={"raw":None,"clean":None,"model":None,"metrics":None,"meta":None,"steps":[],"source":None}

@app.exception_handler(Exception)
async def err(request:Request,exc:Exception):
    return JSONResponse(status_code=500,content={"detail":f"{type(exc).__name__}: {exc}"})

ALIASES={
 "Price":["Price","price","Giá","Gia","gia","selling_price","sale_price","house_price","property_price","amount","cost","value","total_price","gia_ban","giaban"],
 "PricePerM2":["PricePerM2","price_per_m2","price/m2","price/m²","Giá/m2","Giá/m²","Gia/m2","Gia/m²","Giá trên m2","Giá trên m²","don_gia","đơn giá","don_gia_m2","unit_price","price_sqm","price_per_sqm"],
 "Area":["Area","area","Diện tích","DienTich","dien_tich","Area_m2","square","sqm","m2","size","acreage","square_meter","square_meters","surface","floor_area","land_area","dien_tich_m2"],
 "District":["District","district","Quận","Quan","quan_huyen","DistrictName","district_name","quan","borough","county","districtname"],
 "City":["City","city","Thành phố","ThanhPho","thanh_pho","municipality","town"],
 "Province":["Province","province","Tỉnh","Tinh","tinh_thanh","province_name"],
 "Location":["Location","location","Vị trí","ViTri","vi_tri","place","area_name"],
 "PostType":["PostType","post_type","Loại hình","LoaiHinh","PropertyType","property_type","type","house_type","real_estate_type","category"],
 "Bedrooms":["Bedrooms","bedrooms","Phòng ngủ","PhongNgu","bedroom","bed","beds","so_phong_ngu","sophongngu"],
 "Bathrooms":["Bathrooms","bathrooms","Phòng tắm","PhongTam","bathroom","bath","baths","toilet","toilets","wc","so_phong_tam"],
 "Floors":["Floors","floors","Số tầng","SoTang","floor","story","stories","storey","storeys","so_tang"],
 "Direction":["Direction","direction","Hướng","Huong","orientation","house_direction"],
 "Legal":["Legal","legal","Pháp lý","PhapLy","legal_status","legality","ownership"],
 "Interior":["Interior","interior","Nội thất","NoiThat","furniture","furnishing","furnished"],
 "Address":["Address","address","Địa chỉ","DiaChi","location","street_address"],
 "Width_meters":["Width_meters","width","Chiều rộng","ChieuRong","frontage","road_width","width_m"],
 "Entrancewidth":["Entrancewidth","entrance_width","Đường vào","DuongVao","access_width","alley_width"],
 "Ward":["Ward","ward","Phường","Phuong","phuong","commune","xa","Xã"],
 "Project":["Project","project","Dự án","DuAn","du_an","project_name"],
 "Investor":["Investor","investor","Chủ đầu tư","ChuDauTu","developer","builder"],
 "Balcony":["Balcony","balcony","Ban công","BanCong","balcony_direction"],
 "Month":["Month","month","Tháng","Thang","posting_month"],
 "ID":["id","ID","_id","ma","mã","code","record_id","listing_id","property_id"]
}
def _norm_name(x):
    import unicodedata
    z=unicodedata.normalize("NFKD",str(x))
    z="".join(ch for ch in z if not unicodedata.combining(ch)).lower()
    return re.sub(r"[^a-z0-9]+","_",z).strip("_")

def resolve(cols):
    """Exact + normalized + token/fuzzy alias mapping."""
    import difflib
    result={}; used=set()
    for canon,arr in ALIASES.items():
        best=None; bestscore=0
        targets=[_norm_name(x) for x in arr+[canon]]
        for c in cols:
            if c in used: continue
            nc=_norm_name(c)
            score=max(difflib.SequenceMatcher(None,nc,t).ratio() for t in targets)
            if nc in targets: score=1.0
            if score>bestscore: best,bestscore=c,score
        if best is not None and bestscore>=.78:
            result[canon]=best;used.add(best)
    return result

def column_intelligence(raw, mapping):
    """Infer semantic/data role using name + dtype + distribution + sample values."""
    import difflib
    rows=[]
    reverse={v:k for k,v in mapping.items()}
    for c in raw.columns:
        ser=raw[c]; n=len(ser); non=ser.dropna()
        parsed=ser.apply(parse_num); parse_ratio=float(parsed.notna().sum()/max(1,ser.notna().sum()))
        nun=int(ser.nunique(dropna=True)); unique_ratio=nun/max(1,n)
        semantic=reverse.get(c)
        conf=98 if semantic else 0
        nc=_norm_name(c)
        if not semantic:
            bestrole=None;best=.0
            for role,als in ALIASES.items():
                sim=max(difflib.SequenceMatcher(None,nc,_norm_name(a)).ratio() for a in als+[role])
                if sim>best:bestrole,best=role,sim
            # Distribution clues are supporting evidence, not sole semantic assignment.
            if best>=.68:
                semantic=bestrole;conf=int(best*80)
                if parse_ratio>.9 and bestrole in ["Price","Area","Bedrooms","Bathrooms","Floors","Month","ID"]:conf+=10
        if semantic=="ID" or (unique_ratio>.98 and any(k in nc for k in ["id","code","ma_","record"])):
            role="ID";use="Loại khỏi mô hình"
        elif parse_ratio>=.8:
            role="Numeric";use="Numeric predictor"
        elif nun<=min(150,max(20,int(n*.2))):
            role="Categorical";use="One-Hot Encoding"
        else:
            role="Text/high-cardinality";use="Không tự đưa vào X"
        vals=parsed.dropna()
        median=float(vals.median()) if len(vals) else None
        rows.append({"column":str(c),"semantic":semantic or "Không xác định","data_role":role,
                     "confidence":min(conf,100) if semantic else 0,"parse_ratio":round(parse_ratio*100,1),
                     "missing":int(ser.isna().sum()),"unique":nun,"median":median,"model_action":use})
    return rows

def read_csv(b):
    for enc in ("utf-8-sig","utf-8","cp1258","latin1"):
      for sep in (None,",",";","\t"):
        try:
          text=b.decode(enc)
          kw={"sep":sep} if sep else {"sep":None,"engine":"python"}
          d=pd.read_csv(io.StringIO(text),**kw)
          if d.shape[1]>=2:return d
        except: pass
    raise ValueError("Không đọc được CSV.")

def parse_num(x):
    """Parse flexible numeric text incl. Vietnamese thousands/decimal separators and units."""
    if x is None or x is pd.NA:
        return np.nan
    try:
        if pd.isna(x): return np.nan
    except: pass
    if isinstance(x,(int,float,np.integer,np.floating)):
        return float(x) if np.isfinite(x) else np.nan
    z=str(x).strip().lower()
    z=re.sub(r'(?i)(m²|m2|sqm|sq\.?m|mét vuông|meters?|metres?|phòng|tầng|bedrooms?|bathrooms?)','',z).strip()
    m=re.search(r'-?[\d.,]+',z)
    if not m:return np.nan
    t=m.group(0)
    # 1.234,56 => 1234.56 ; 1,234.56 => 1234.56
    if "," in t and "." in t:
        if t.rfind(",")>t.rfind("."): t=t.replace(".","").replace(",",".")
        else: t=t.replace(",","")
    elif t.count(".")>1: t=t.replace(".","")
    elif t.count(",")>1: t=t.replace(",","")
    elif "," in t:
        a,b=t.split(",",1)
        t=a+b if len(b)==3 and len(a)>=1 else a+"."+b
    elif "." in t:
        a,b=t.split(".",1)
        # keep decimal dot for small physical measurements; thousands-like text gets collapsed
        if len(b)==3 and len(a)>=1 and abs(float(a or 0))>=1: t=a+b
    try:return float(t)
    except:return np.nan

def infer_price_unit(series):
    """Infer unit for unitless price values. Output conversion factor to billion VND."""
    vals=series.dropna().astype(float)
    vals=vals[(vals>0)&np.isfinite(vals)]
    if len(vals)==0:return ("unknown",1.0,"Không đủ dữ liệu")
    med=float(vals.median()); q25=float(vals.quantile(.25)); q75=float(vals.quantile(.75))
    # User-requested heuristic: predominantly 1000–10000 => million; 1–100 => billion.
    share_million=float(vals.between(1000,10000).mean())
    share_billion=float(vals.between(1,100).mean())
    if share_million>=.5 or med>=500:
        return ("triệu",.001,f"Trung vị={med:.2f}; {share_million*100:.1f}% nằm trong 1.000–10.000 → suy luận đơn vị triệu")
    if share_billion>=.5 or med<=100:
        return ("tỷ",1.0,f"Trung vị={med:.2f}; {share_billion*100:.1f}% nằm trong 1–100 → suy luận đơn vị tỷ")
    # ambiguous middle range: choose million only when magnitude strongly suggests it
    if med>100:return ("triệu",.001,f"Trung vị={med:.2f} >100 → suy luận đơn vị triệu")
    return ("tỷ",1.0,f"Trung vị={med:.2f} → suy luận đơn vị tỷ")

def infer_physical_unit(name, series):
    """Infer common physical unit conversions from name + magnitude."""
    n=str(name).lower(); vals=series.dropna().astype(float)
    if len(vals)==0:return (1.0,"giữ nguyên")
    med=float(vals.median())
    if any(k in n for k in ["area","diện tích","dien_tich","size","square"]):
        # cm² is unrealistic for housing CSVs; support m² and common hectare magnitude/name.
        if any(k in n for k in ["ha","hectare"]) or (0<med<1 and vals.quantile(.9)<=10):
            return (10000.0,"ha → m²")
        return (1.0,"m²/giữ nguyên")
    if any(k in n for k in ["width","length","distance","chiều","rong","rộng"]):
        if any(k in n for k in ["cm","centimeter"]) or med>100:
            return (.01,"cm → m")
        return (1.0,"m/giữ nguyên")
    return (1.0,"giữ nguyên")

def cat_legal(x):
    s="" if pd.isna(x) else str(x); z=s.lower()
    if not z or z=="0": return "Chưa rõ"
    if "sổ đỏ" in z or "sổ hồng" in z or "có sổ" in z:return "Sổ đỏ/Sổ hồng"
    if "hợp đồng mua bán" in z or "hđmb" in z:return "HĐMB"
    if "chờ sổ" in z:return "Đang chờ sổ"
    return "Khác"
def cat_interior(x):
    s="" if pd.isna(x) else str(x); z=s.lower()
    if not z or z=="0":return "Chưa rõ"
    if "không nội thất" in z:return "Không nội thất"
    if any(k in z for k in ["cao cấp","đầy đủ","full"]):return "Đầy đủ/Cao cấp"
    if any(k in z for k in ["cơ bản","hoàn thiện"]):return "Cơ bản"
    return "Khác"

HANOI=['Ba Đình','Hoàn Kiếm','Tây Hồ','Long Biên','Cầu Giấy','Đống Đa','Hai Bà Trưng','Hoàng Mai','Thanh Xuân','Nam Từ Liêm','Bắc Từ Liêm','Hà Đông','Sơn Tây','Ba Vì','Chương Mỹ','Đan Phượng','Đông Anh','Gia Lâm','Hoài Đức','Mê Linh','Mỹ Đức','Phú Xuyên','Phúc Thọ','Quốc Oai','Sóc Sơn','Thạch Thất','Thanh Oai','Thanh Trì','Thường Tín','Ứng Hòa']
RES=['Nhà riêng','Căn hộ chung cư','Nhà biệt thự, liền kề','Nhà mặt phố','Shophouse, nhà phố thương mại','Chung cư mini, căn hộ dịch vụ','Condotel']

def snap(title,before,after,note,table=None):
    return {"title":title,"before":int(before),"after":int(after),"removed":int(before-after),"note":note,"table":table or []}


HANOI_CANON={
"ba đình","bắc từ liêm","chương mỹ","cầu giấy","gia lâm","hai bà trưng","hoài đức",
"hoàn kiếm","hoàng mai","hà đông","long biên","mê linh","nam từ liêm","quốc oai",
"sóc sơn","sơn tây","thanh oai","thanh trì","thanh xuân","thường tín","thạch thất",
"tây hồ","đan phượng","đông anh","đống đa","ba vi","ba vì","mỹ đức","phúc thọ",
"phú xuyên","ứng hòa"
}
RESID_CANON={
"căn hộ chung cư","chung cư mini, căn hộ dịch vụ","nhà biệt thự, liền kề",
"nhà mặt phố","nhà riêng","shophouse, nhà phố thương mại","condotel"
}
def norm_text(v):
    import unicodedata
    z=unicodedata.normalize("NFKD",str(v))
    z="".join(ch for ch in z if not unicodedata.combining(ch)).lower()
    return re.sub(r"\\s+"," ",z.strip())

HANOI_CANON_NORM=None
RESID_CANON_NORM=None
def _scope_sets():
    global HANOI_CANON_NORM,RESID_CANON_NORM
    if HANOI_CANON_NORM is None:
        HANOI_CANON_NORM={norm_text(x) for x in HANOI_CANON}
        RESID_CANON_NORM={norm_text(x) for x in RESID_CANON}
    return HANOI_CANON_NORM,RESID_CANON_NORM

def rare_group(series,min_count=10):
    vc=series.value_counts(dropna=False)
    rare=set(vc[vc<min_count].index)
    return series.apply(lambda x:"Khác" if x in rare else x), len(rare)


HANOI_TOKENS = {
    "ha noi","hanoi",
    "ba dinh","bac tu liem","chuong my","cau giay","gia lam","hai ba trung","hoai duc",
    "hoan kiem","hoang mai","ha dong","long bien","me linh","nam tu liem","quoc oai",
    "soc son","son tay","thanh oai","thanh tri","thanh xuan","thuong tin","thach that",
    "tay ho","dan phuong","dong anh","dong da","ba vi","my duc","phuc tho","phu xuyen","ung hoa"
}
def _geo_clean(v):
    if pd.isna(v): return ""
    z=norm_text(v)
    z=re.sub(r"\b(quan|huyen|thi xa|thanh pho|tp|district|city|province)\b"," ",z)
    return re.sub(r"\s+"," ",z).strip()

def _is_hanoi_value(v):
    z=_geo_clean(v)
    if not z:return False
    return any(tok==z or tok in z for tok in HANOI_TOKENS)

def hanoi_scope_mask(df):
    """Strict Hanoi scope with flexible schema discovery.
    Returns mask, evidence columns, and confidence note.
    Positive evidence from any recognized geographic field keeps a row.
    """
    geo=[]
    for c in ["District","City","Province","Address","Location","Ward"]:
        if c in df.columns: geo.append(c)
    # Also inspect unrecognized columns whose names clearly imply geography.
    for c in df.columns:
        nc=_norm_name(c)
        if c not in geo and any(k in nc for k in ["district","quan","huyen","city","province","tinh","address","location","ward","phuong","commune"]):
            geo.append(c)
    geo=list(dict.fromkeys(geo))
    if not geo:
        return pd.Series(False,index=df.index),[], "Không tìm thấy cột địa lý"
    mask=pd.Series(False,index=df.index)
    for c in geo:
        mask = mask | df[c].apply(_is_hanoi_value)
    return mask,geo,f"Dựa trên: {', '.join(map(str,geo))}"
NOTEBOOK_REQUIRED = ['Address','District','Price','Area','Direction','Bedrooms',
    'Bathrooms','Floors','Width_meters','Legal','Interior','Entrancewidth','PostType']

NOTEBOOK_HANOI = ['Ba Đình','Hoàn Kiếm','Tây Hồ','Long Biên','Cầu Giấy','Đống Đa',
    'Hai Bà Trưng','Hoàng Mai','Thanh Xuân','Nam Từ Liêm','Bắc Từ Liêm','Hà Đông',
    'Sơn Tây','Ba Vì','Chương Mỹ','Đan Phượng','Đông Anh','Gia Lâm','Hoài Đức',
    'Mê Linh','Mỹ Đức','Phú Xuyên','Phúc Thọ','Quốc Oai','Sóc Sơn','Thạch Thất',
    'Thanh Oai','Thanh Trì','Thường Tín','Ứng Hòa']

NOTEBOOK_RESIDENTIAL = ['Nhà riêng','Căn hộ chung cư','Nhà biệt thự, liền kề','Nhà mặt phố',
    'Shophouse, nhà phố thương mại','Chung cư mini, căn hộ dịch vụ','Condotel']

def is_notebook_schema(raw):
    # Exact-mode only when the original notebook identity columns are physically present.
    # Alias/fuzzy matching is intentionally NOT enough to activate destructive subset dedupe.
    return all(c in raw.columns for c in NOTEBOOK_REQUIRED)

def preprocess_notebook_exact(raw):
    """Exact preprocessing reproduced from Phan_tich_du_bao_gia_nha_HaNoi(3).ipynb."""
    d=raw.copy(); steps=[]
    def rec(name,before,detail="",mode="notebook_exact"):
        after=len(d); steps.append({"name":name,"before":int(before),"after":int(after),
            "removed":int(before-after),"removed_pct":round((before-after)*100/max(1,before),2),
            "detail":detail,"mode":mode})

    # 2.1 duplicate identity: exact notebook subset
    before=len(d)
    d=d.drop_duplicates(subset=NOTEBOOK_REQUIRED,keep='first').reset_index(drop=True)
    rec("2.1 Loại tin đăng trùng",before,
        "Notebook exact: 13 cột nhận dạng Address, District, Price, Area, Direction, Bedrooms, Bathrooms, Floors, Width_meters, Legal, Interior, Entrancewidth, PostType")

    def pnum(x):
        if pd.isna(x): return np.nan
        x=str(x).strip().replace(',','.')
        try:return float(x)
        except ValueError:return np.nan

    # 2.2 Area and Price exactly as notebook
    d["Area_m2"]=d["Area"].astype(str).str.replace(" m²","",regex=False).apply(pnum)
    def pprice(row):
        p=str(row["Price"]).strip(); area=row["Area_m2"]
        if "tỷ" in p:return pnum(p.replace("tỷ","").strip())
        if "triệu/m²" in p:
            z=pnum(p.replace("triệu/m²","").strip())
            return z*area/1000.0 if pd.notna(z) and pd.notna(area) else np.nan
        if "triệu/tháng" in p:return np.nan
        if "triệu" in p:
            z=pnum(p.replace("triệu","").strip())
            return z/1000.0 if pd.notna(z) else np.nan
        return np.nan
    d["Price_ty"]=d.apply(pprice,axis=1)

    # 2.3 rooms/floors
    def proom(x):
        x=str(x).strip()
        if x in ("0","nan"):return np.nan
        return pnum(x.replace("phòng","").strip())
    def pfloor(x):
        x=str(x).strip()
        if x in ("0","nan"):return np.nan
        return pnum(x.replace("tầng","").strip())
    d["Bedrooms_n"]=d["Bedrooms"].apply(proom)
    d["Bathrooms_n"]=d["Bathrooms"].apply(proom)
    d["Floors_n"]=d["Floors"].apply(pfloor)

    # 2.4 categories
    d["PropertyType"]=d["PostType"].str.split(" tại ").str[0]
    def cat_legal(x):
        x=str(x)
        if x=="0":return "Chưa rõ"
        xl=x.lower()
        if "sổ đỏ" in xl or "sổ hồng" in xl or "có sổ" in xl:return "Sổ đỏ/Sổ hồng"
        if "hợp đồng mua bán" in xl or "hđmb" in xl:return "HĐMB"
        if "đang chờ sổ" in xl:return "Đang chờ sổ"
        return "Khác"
    def cat_interior(x):
        x=str(x)
        if x=="0":return "Chưa rõ"
        xl=x.lower()
        if "không nội thất" in xl:return "Không nội thất"
        if "cao cấp" in xl or "đầy đủ" in xl or "full" in xl:return "Đầy đủ/Cao cấp"
        if "cơ bản" in xl or "hoàn thiện" in xl:return "Cơ bản"
        return "Khác"
    d["Legal_cat"]=d["Legal"].apply(cat_legal)
    d["Interior_cat"]=d["Interior"].apply(cat_interior)
    d["Direction_cat"]=d["Direction"].replace("0","Chưa rõ")

    # 2.5 exact scope
    before=len(d)
    d=d[d["District"].isin(NOTEBOOK_HANOI)]
    d=d[d["PropertyType"].isin(NOTEBOOK_RESIDENTIAL)]
    d=d.dropna(subset=["Price_ty","Area_m2"]).copy()
    rec("2.5 Lọc phạm vi phân tích",before,"Notebook exact: Hà Nội + BĐS để ở + bắt buộc Price_ty và Area_m2")

    # 2.5b rare grouping
    counts=d["PropertyType"].value_counts()
    rare=counts[counts<10].index.tolist()
    d["PropertyType"]=d["PropertyType"].replace({t:"Chung cư mini, căn hộ dịch vụ" for t in rare})

    # 2.6 missing exact
    before=len(d)
    apt=["Căn hộ chung cư","Chung cư mini, căn hộ dịch vụ","Condotel"]
    d.loc[d["PropertyType"].isin(apt)&d["Floors_n"].isna(),"Floors_n"]=1
    for c in ["Bedrooms_n","Bathrooms_n","Floors_n"]:
        d[c]=d.groupby("PropertyType")[c].transform(lambda x:x.fillna(x.median()))
        d[c]=d[c].fillna(d[c].median())
    rec("2.6 Xử lý giá trị khuyết",before,"Median theo PropertyType; căn hộ thiếu Floors_n = 1")

    # 2.7 sequential IQR exactly
    n0=len(d)
    for c in ["Price_ty","Area_m2","Bedrooms_n","Bathrooms_n","Floors_n"]:
        q1,q3=d[c].quantile(.25),d[c].quantile(.75); iqr=q3-q1
        lo=max(q1-3*iqr,0); hi=q3+3*iqr; before=len(d)
        d=d[(d[c]>=lo)&(d[c]<=hi)].copy()
        rec("2.7 IQR k=3: "+c,before,f"Notebook exact; ngưỡng [{lo:.2f}, {hi:.2f}]","notebook_exact")

    before=len(d)
    d=d[(d["Area_m2"]<=1000)&(d["Bedrooms_n"]<=20)&(d["Bathrooms_n"]<=20)&(d["Floors_n"]<=20)].copy()
    rec("2.7 Giới hạn nghiệp vụ",before,"Area≤1000; Bedrooms/Bathrooms/Floors≤20","notebook_exact")
    before=len(d); d=d[d["Price_ty"]>=.3].copy()
    rec("2.7 Price ≥ 0.3 tỷ",before,"Notebook exact","notebook_exact")

    # 2.8 derived
    d["Price_per_m2"]=d["Price_ty"]*1000/d["Area_m2"]
    d["log_Price"]=np.log(d["Price_ty"])
    d["log_Area"]=np.log(d["Area_m2"])
    steps.append({"name":"Notebook Exact hoàn tất","before":n0,"after":len(d),
        "removed":n0-len(d),"removed_pct":round((n0-len(d))*100/max(1,n0),1),
        "detail":"Pipeline 2.1–2.8 từ notebook hiện hành","mode":"notebook_exact"})
    return d.reset_index(drop=True),steps

def preprocess_flexible(raw):
    d=raw.copy(); steps=[]
    def rec(name,before,detail="",mode=""):
        after=len(d);steps.append({"name":name,"before":int(before),"after":int(after),"removed":int(before-after),
        "removed_pct":round((before-after)*100/max(1,before),2),"detail":detail,"mode":mode})
    mp=resolve(d.columns);d=d.rename(columns={v:k for k,v in mp.items() if k!="ID"})
    before=len(d); d=d.drop_duplicates()
    rec("Loại bản ghi trùng",before,"Flexible mode: chỉ loại dòng trùng toàn bộ; không áp subset notebook cho schema lạ","safe")
    area0=d["Area"].apply(parse_num) if "Area" in d.columns else None

    def cv(v,a=None):
        if pd.isna(v):return np.nan
        t=str(v).lower();nt=norm_text(t)
        if "/thang" in nt or "thoa thuan" in nt:return np.nan
        z=parse_num(v)
        if pd.isna(z):return np.nan
        if "trieu/m" in nt:return z*a/1000 if a is not None and pd.notna(a) and a>0 else np.nan
        if "ty" in nt:return z
        if "trieu" in nt:return z/1000
        if z>=1e7:return z/1e9
        return z*factor

    before=len(d)
    if "Price" in d.columns:
        unit,factor,reason=infer_price_unit(d["Price"].apply(parse_num))
        d["Price_ty"]=[cv(v,area0.iloc[i] if area0 is not None else None) for i,v in enumerate(d["Price"])]
        price_detail=f"Tổng giá → tỷ VNĐ; {unit}: {reason}"
    elif "PricePerM2" in d.columns and "Area" in d.columns:
        # Giá/m² is normally million VND/m². Parse textual and numeric forms.
        def unit_price_million(v):
            if pd.isna(v): return np.nan
            z=parse_num(v)
            if pd.isna(z): return np.nan
            nt=norm_text(str(v))
            if "ty/m" in nt: return z*1000
            if "trieu/m" in nt: return z
            if z>=1e6: return z/1e6   # raw VND/m²
            return z                 # numeric 86.96 => million/m²
        up=d["PricePerM2"].apply(unit_price_million)
        d["Price_ty"]=up*area0/1000
        price_detail="Không có Tổng giá: suy ra Tổng giá = Giá/m² × Diện tích; quy đổi về tỷ VNĐ"
    else:
        raise ValueError("Không nhận diện được Tổng giá. Cần cột Tổng giá, hoặc đồng thời cột Giá/m² và Diện tích.")

    d=d[d["Price_ty"].notna()&(d["Price_ty"]>0)].copy()
    rec("Chuẩn hóa giá",before,price_detail,"semantic")
    for a,b in [("Area","Area_m2"),("Bedrooms","Bedrooms_n"),("Bathrooms","Bathrooms_n"),("Floors","Floors_n")]:
        if a in d.columns:
            z=d[a].apply(parse_num)
            if a=="Area":z=z*infer_physical_unit(a,z)[0]
            d[b]=z; d.loc[d[b]<=0,b]=np.nan
    for c in ["District","PostType","Direction","Legal","Interior","Ward","Project","Investor","Balcony"]:
        if c in d.columns:d[c]=d[c].astype("string").str.strip().fillna("Chưa rõ").replace({"":"Chưa rõ"})
    if "PostType" in d.columns:d["PropertyType"]=d["PostType"].astype(str)
    for a,b in [("Legal","Legal_cat"),("Interior","Interior_cat"),("Direction","Direction_cat")]:
        if a in d.columns:d[b]=d[a].astype(str)
    before=len(d);hmask,geo_cols,geo_note=hanoi_scope_mask(d)
    if not geo_cols:raise ValueError("Không xác định được phạm vi Hà Nội: file không có cột địa lý đủ tin cậy.")
    matched=int(hmask.sum())
    if matched<50:raise ValueError(f"Chỉ nhận diện được {matched} bản ghi thuộc Hà Nội. Kiểm tra tên cột/giá trị địa lý hoặc bổ sung alias.")
    d=d[hmask].copy();rec("Lọc bắt buộc phạm vi Hà Nội",before,f"{geo_note}; giữ {matched}/{before} dòng","scope")
    before=len(d)
    for c in ["Bedrooms_n","Bathrooms_n","Floors_n"]:
        if c in d.columns:d[c]=d[c].fillna(d[c].median())
    rec("Xử lý missing biến lõi",before,"Median; không xóa dòng","impute")
    for c in ["Price_ty","Area_m2","Bedrooms_n","Bathrooms_n","Floors_n"]:
        if c not in d.columns:continue
        z=d[c].dropna()
        if len(z)<20:continue
        q1,q3=z.quantile(.25),z.quantile(.75);iqr=q3-q1
        if not np.isfinite(iqr) or iqr<=0:continue
        lo=max(q1-3*iqr,0);hi=q3+3*iqr;bad=(d[c]<lo)|(d[c]>hi);frac=float(bad.mean());before=len(d)
        if frac<=.15:d=d[~bad].copy();rec("IQR k=3: "+c,before,f"{frac:.1%} bị loại","remove")
        else:
            if c!="Price_ty":d[c]=d[c].clip(lo,hi)
            steps.append({"name":"IQR k=3: "+c,"before":before,"after":before,"removed":0,"removed_pct":0,"detail":f"{frac:.1%} → Safety Guard","mode":"guard"})
    checks=[]
    if "Area_m2" in d.columns:checks.append(("Area ≤ 1000",lambda x:(x["Area_m2"]>0)&(x["Area_m2"]<=1000)))
    if "Bedrooms_n" in d.columns:checks.append(("Bedrooms ≤ 20",lambda x:x["Bedrooms_n"].between(0,20)))
    if "Bathrooms_n" in d.columns:checks.append(("Bathrooms ≤ 20",lambda x:x["Bathrooms_n"].between(0,20)))
    if "Floors_n" in d.columns:checks.append(("Floors 1–20",lambda x:x["Floors_n"].between(1,20)))
    checks.append(("Price ≥ 0.3 tỷ",lambda x:x["Price_ty"]>=.3))
    for name,make_mask in checks:
        mask=make_mask(d).fillna(False)
        frac=float((~mask).mean());before=len(d)
        if frac<=.15:d=d.loc[mask].copy();rec(name,before,f"{frac:.1%} vi phạm","business")
        else:steps.append({"name":name,"before":before,"after":before,"removed":0,"removed_pct":0,"detail":f"{frac:.1%} → Safety Guard","mode":"guard"})
    if "Area_m2" in d.columns:
        d["Price_per_m2"]=d["Price_ty"]*1000/d["Area_m2"];ok=d["Price_per_m2"].between(10,1500);frac=float((~ok.fillna(False)).mean());before=len(d)
        if frac<=.15:d=d[ok.fillna(False)].copy();rec("Đơn giá 10–1500 tr/m²",before,f"{frac:.1%} vi phạm","business")
        else:steps.append({"name":"Đơn giá 10–1500 tr/m²","before":before,"after":before,"removed":0,"removed_pct":0,"detail":f"{frac:.1%} → nghi đơn vị/schema, giữ dữ liệu","mode":"guard"})
        d["log_Area"]=np.log(d["Area_m2"].clip(lower=1e-9))
    d["log_Price"]=np.log(d["Price_ty"].clip(lower=1e-9))
    if len(d)<50:raise ValueError("Dữ liệu sau xử lý quá ít.")
    return d.reset_index(drop=True),steps

def preprocess(raw):
    return preprocess_notebook_exact(raw) if is_notebook_schema(raw) else preprocess_flexible(raw)

def train(d):
    target="log_Price"
    # Leakage/raw target representations are never predictors.
    leakage={"Price","PricePerM2","Price_ty","Price_per_m2","log_Price"}
    numeric=[]; categorical=[]; ignored=[]
    for c in d.columns:
        if c in leakage: continue
        ss=d[c]
        nun=int(ss.nunique(dropna=True))
        if nun<=1:
            ignored.append({"column":c,"reason":"Cột hằng"}); continue
        if pd.api.types.is_numeric_dtype(ss):
            # Near-unique integer/numeric identifiers are excluded.
            if nun/len(d)>.98 and any(k in str(c).lower() for k in ["id","ma_","mã","code","index"]):
                ignored.append({"column":c,"reason":"Có dấu hiệu là ID"}); continue
            numeric.append(c)
        else:
            # Flexible categorical handling. High-cardinality free text is excluded, not fatal.
            if nun<=min(150,max(20,int(len(d)*.20))):
                categorical.append(c)
            else:
                ignored.append({"column":c,"reason":f"Text/cardinality cao ({nun} mức)"})
    use=numeric+categorical
    if not use: raise ValueError("Không có biến độc lập phù hợp để xây dựng hồi quy đa biến.")
    X=pd.get_dummies(d[use],columns=categorical,drop_first=True,dummy_na=False).astype(float)
    # Remove zero-variance post-encoding columns
    X=X.loc[:,X.nunique()>1]
    if X.shape[1]<2: raise ValueError("Sau tiền xử lý còn dưới 2 biến độc lập; chưa đủ cho hồi quy tuyến tính đa biến.")
    y=d[target]
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
    lr=LinearRegression().fit(Xtr,ytr); pl=lr.predict(Xte); ptr=lr.predict(Xtr)
    smear=float(np.mean(np.exp(ytr-ptr))); pred=np.exp(pl)*smear; true=np.exp(yte)
    m={"r2_log":float(r2_score(yte,pl)),"r2_price":float(r2_score(true,pred)),
       "mae":float(mean_absolute_error(true,pred)),"rmse":float(mean_squared_error(true,pred)**.5),
       "mape":float(np.mean(np.abs((true-pred)/true))*100),"smear":smear,
       "sigma":float(np.sqrt(np.mean((ytr-ptr)**2))),"train":len(Xtr),"test":len(Xte)}
    return lr,X.columns.tolist(),m,{"numeric":numeric,"categorical":categorical,"ignored":ignored,
        "used_source_columns":use,"source_columns":len(d.columns)}

def payload():
    d=STATE["clean"]; m=STATE["metrics"]
    return {"source":STATE["source"],"raw_rows":len(STATE["raw"]),"raw_cols":len(STATE["raw"].columns),
      "clean_rows":len(d),"clean_cols":len(d.columns),"steps":STATE["steps"],"metrics":m,
      "levels":{c:sorted(d[c].astype(str).unique().tolist()) for c in ["District","PropertyType","Legal_cat","Interior_cat","Direction_cat"] if c in d.columns},
      "medians":{c:float(d[c].median()) for c in ["Area_m2","Bedrooms_n","Bathrooms_n","Floors_n"] if c in d.columns},
      "feature_meta":STATE["meta"], "intelligence":STATE["meta"].get("intelligence",[]),
      "retention":{"raw":len(STATE["raw"]),"clean":len(d),"kept_pct":round(len(d)*100/max(1,len(STATE["raw"])),2),"removed":len(STATE["raw"])-len(d)},
      "preview":d.head(10).replace({np.nan:None}).to_dict("records")}

def load_dataset_into_state(raw, source):
    """Use the original V8.5.2 auto-normalization pipeline, then train and store current state."""
    intelligence=column_intelligence(raw,resolve(raw.columns))
    clean,steps=preprocess(raw)
    lr,cols,m,feature_meta=train(clean)
    STATE.update(raw=raw,clean=clean,steps=steps,model=lr,metrics=m,
                 meta={"columns":cols,**feature_meta,"intelligence":intelligence},source=source)
    clean.to_csv(DATA/"HN_Houseprice_cleaned.csv",index=False)
    joblib.dump(STATE,MODELS/"current.joblib")
    return payload()

@app.on_event("startup")
def load_default_dataset():
    """Start ready-to-predict with the notebook dataset; upload remains optional."""
    p=DATA/"HN_Houseprice_default.csv"
    if not p.exists():
        raise RuntimeError("Thiếu data/HN_Houseprice_default.csv")
    raw=pd.read_csv(p)
    load_dataset_into_state(raw,"HN_Houseprice_default.csv")

@app.get("/")
def home():return FileResponse(BASE/"static"/"index.html")

@app.get("/api/state")
def state():
    if STATE["clean"] is None:
        raise HTTPException(409,"Dữ liệu mặc định chưa sẵn sàng")
    return payload()

@app.post("/api/reset-default")
def reset_default():
    p=DATA/"HN_Houseprice_default.csv"
    raw=pd.read_csv(p)
    return load_dataset_into_state(raw,"HN_Houseprice_default.csv")

@app.post("/api/upload")
async def upload(file:UploadFile=File(...)):
    """Upload only previews the CSV. It does NOT train until the user confirms."""
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400,"Chỉ hỗ trợ tệp CSV")
    raw_bytes=await file.read()
    try:
        raw=pd.read_csv(io.BytesIO(raw_bytes))
    except Exception:
        try:
            raw=pd.read_csv(io.BytesIO(raw_bytes),encoding="utf-8-sig")
        except Exception as e:
            raise HTTPException(400,f"Không đọc được CSV: {e}")
    if raw.empty:
        raise HTTPException(400,"CSV không có dữ liệu")

    # Save pending file separately; current default/model remains active.
    pending=DATA/"pending_upload.csv"
    raw.to_csv(pending,index=False)
    mapping=resolve(raw.columns)
    intel=column_intelligence(raw,mapping)

    suggestions=[]
    for c in raw.columns:
        name=str(c)
        ser=raw[c]
        miss=float(ser.isna().mean())
        nun=int(ser.nunique(dropna=True))
        if re.match(r"^Unnamed:\s*\d+$",name,re.I):
            suggestions.append({"level":"warning","column":name,
                "message":"Có thể là cột chỉ số được tạo khi xuất CSV. Hãy kiểm tra trước khi giữ hoặc xóa."})
        if miss>=0.50:
            suggestions.append({"level":"warning","column":name,
                "message":f"Cột có {miss:.1%} giá trị khuyết."})
        if len(raw)>20 and nun==len(raw):
            suggestions.append({"level":"info","column":name,
                "message":"Mỗi dòng gần như có một giá trị riêng; có thể là ID hoặc trường định danh."})

    # Semantic recognition suggestions from V8.5.2 engine.
    for sem,col in mapping.items():
        if col:
            suggestions.append({"level":"ok","column":col,
                "message":f"Hệ thống dự kiến nhận diện cột này là “{sem}” khi xử lý."})

    preview=raw.head(20).copy()
    preview=preview.where(pd.notna(preview),None)
    return {
        "pending":True,
        "filename":file.filename,
        "rows":int(len(raw)),
        "columns":[str(x) for x in raw.columns],
        "preview":preview.to_dict(orient="records"),
        "suggestions":suggestions[:80],
        "intelligence":intel,
        "message":"Tệp mới chỉ được nạp để kiểm tra. Mô hình hiện tại chưa bị thay đổi."
    }

@app.post("/api/confirm-upload")
async def confirm_upload(request:Request):
    """Accept edited preview/full data and then run the original V8.5.2 pipeline."""
    body=await request.json()
    pending=DATA/"pending_upload.csv"
    if not pending.exists():
        raise HTTPException(409,"Chưa có CSV đang chờ kiểm tra")
    raw=pd.read_csv(pending)

    # Apply optional column renames requested in the review UI.
    renames=body.get("renames") or {}
    renames={str(k):str(v).strip() for k,v in renames.items()
             if str(v).strip() and str(k) in raw.columns and str(v).strip()!=str(k)}
    if len(set(renames.values())) != len(renames.values()):
        raise HTTPException(400,"Tên cột sau khi sửa bị trùng nhau")
    raw=raw.rename(columns=renames)

    # Delete optional columns selected by the user.
    drops=[c for c in (body.get("drop_columns") or []) if c in raw.columns]
    if drops:
        raw=raw.drop(columns=drops)

    # Apply edits from the visible preview rows. Other rows remain untouched.
    edits=body.get("edits") or []
    for e in edits:
        try:
            row=int(e.get("row"))
            col=str(e.get("column"))
            if row in raw.index and col in raw.columns:
                val=e.get("value")
                raw.at[row,col]=None if val is None or str(val).strip()=="" else val
        except Exception:
            continue

    filename=str(body.get("filename") or "uploaded_reviewed.csv")
    return load_dataset_into_state(raw,filename)

@app.get("/api/pending-csv")
def pending_csv():
    p=DATA/"pending_upload.csv"
    if not p.exists():
        raise HTTPException(404,"Chưa có CSV đang chờ")
    return FileResponse(p,media_type="text/csv",filename="csv_dang_kiem_tra.csv")


@app.get("/api/eda")
def eda():
    if STATE["clean"] is None: raise HTTPException(409,"Chưa có dữ liệu")
    d=STATE["clean"]; price=d[["Price_ty","log_Price"]].dropna()
    def five_num(series):
        z=series.dropna().astype(float)
        if not len(z): return None
        return {"min":float(z.min()),"q1":float(z.quantile(.25)),"median":float(z.median()),
                "q3":float(z.quantile(.75)),"max":float(z.max()),"mean":float(z.mean())}
    out={"price":price.sample(min(2500,len(price)),random_state=42).to_dict("records"),
         "area":[],"scatter":[],"district":[],"types":[],
         "price_stats":five_num(d["Price_ty"]),"area_stats":five_num(d["Area_m2"]) if "Area_m2" in d.columns else None}
    if "Area_m2" in d.columns:
        area=d[["Area_m2"]].dropna(); out["area"]=area.sample(min(2500,len(area)),random_state=42).to_dict("records")
        sc=d[["Area_m2","Price_ty"]].dropna(); out["scatter"]=sc.sample(min(800,len(sc)),random_state=42).to_dict("records")
    corrcols=["log_Price"]+[c for c in STATE["meta"].get("numeric",[]) if c not in ("Price_ty","Price_per_m2","log_Price")][:11]
    corrcols=list(dict.fromkeys([c for c in corrcols if c in d.columns]))
    corr=d[corrcols].corr().round(3); out["corr"]=corr.to_dict(); out["corr_columns"]=corrcols
    if "District" in d.columns:
        out["district"]=(d.groupby("District").agg(median_price=("Price_ty","median"),count=("Price_ty","size"))
                         .sort_values("median_price",ascending=False).reset_index().to_dict("records"))
    if "PropertyType" in d.columns:
        out["types"]=(d.groupby("PropertyType").agg(median_price=("Price_ty","median"),count=("Price_ty","size"))
                      .sort_values("median_price",ascending=False).reset_index().to_dict("records"))
    return out

class P(BaseModel):
    District:str
    PropertyType:str
    Area_m2:float
    Bedrooms_n:float
    Bathrooms_n:float
    Floors_n:float|None=None
    Legal_cat:str
    Interior_cat:str
    Direction_cat:str

@app.post("/api/predict")
def predict(x:P):
    incoming=x.model_dump()
    base={}
    clean=STATE["clean"]
    used=STATE["meta"]["used_source_columns"]

    # Default missing/extra predictors: numeric median, categorical mode.
    for c in used:
        if pd.api.types.is_numeric_dtype(clean[c]):
            base[c]=float(clean[c].median())
        else:
            md=clean[c].mode(dropna=True)
            base[c]=str(md.iloc[0]) if len(md) else "Chưa rõ"

    # User-provided values override defaults, except Floors_n when UI sends null.
    for k,v in incoming.items():
        if k in base and v is not None:
            base[k]=v

    if "Area_m2" in base:
        base["Area_m2"]=incoming["Area_m2"]
    if "log_Area" in base:
        base["log_Area"]=float(np.log(incoming["Area_m2"]))

    # Prediction-only imputation for Floors_n.
    # IMPORTANT: training/notebook cleaning remains unchanged (7,225 rows).
    floor_used=None; floor_source="user"
    if "Floors_n" in base and incoming.get("Floors_n") is None:
        group=clean
        if "PropertyType" in clean.columns and incoming.get("PropertyType"):
            g=clean[clean["PropertyType"].astype(str)==str(incoming["PropertyType"])]
            if len(g) and g["Floors_n"].notna().any():
                group=g
        med=group["Floors_n"].median()
        if pd.isna(med): med=clean["Floors_n"].median()
        base["Floors_n"]=float(med)
        floor_used=float(med); floor_source="median_by_property_type"

    row=pd.DataFrame([base])
    cats=STATE["meta"]["categorical"]
    X=pd.get_dummies(row,columns=[c for c in cats if c in row.columns],drop_first=False).astype(float)
    X=X.reindex(columns=STATE["meta"]["columns"],fill_value=0)
    lg=float(STATE["model"].predict(X)[0]); m=STATE["metrics"]; p=float(np.exp(lg)*m["smear"])

    # Warn on extrapolation rather than silently clipping user input.
    warnings=[]
    for field,label in [("Area_m2","Diện tích"),("Bedrooms_n","Số phòng ngủ"),("Bathrooms_n","Số phòng tắm")]:
        if field in clean.columns and incoming.get(field) is not None:
            lo=float(clean[field].min()); hi=float(clean[field].max()); val=float(incoming[field])
            if val<lo or val>hi:
                warnings.append(f"{label} {val:g} nằm ngoài khoảng dữ liệu huấn luyện [{lo:g}, {hi:g}].")
    return {"price":p,"ppm":p*1000/x.Area_m2,
            "lo":float(np.exp(lg-1.2816*m["sigma"])*m["smear"]),
            "hi":float(np.exp(lg+1.2816*m["sigma"])*m["smear"]),
            "floors_used":floor_used,"floors_source":floor_source,"warnings":warnings}

