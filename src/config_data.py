from __future__ import annotations
import dataclasses, datetime as dt, gc, hashlib, json, os, random, sys, warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional
import numpy as np
import pandas as pd
from PIL import Image
warnings.filterwarnings("ignore", category=UserWarning)

@dataclass
class Config:
    PROJECT_NAME: str = "DFU-ImageGuard"
    REPO_FULL_NAME: str = "AzizulHakim00/DFU-ImageGuard"
    DATASET_HANDLE: str = "laithjj/diabetic-foot-ulcer-dfu"
    DATASET_TITLE: str = "diabetic foot ulcer (DFU)"
    DATASET_LICENSE: str = "Unknown / not declared on the Kaggle data card"
    DATASET_CITATIONS: list[str] = field(default_factory=lambda: [
        "Alzubaidi et al. (2020), DFU_QUTNet, Multimedia Tools and Applications 79, 15655–15677.",
        "Alzubaidi et al. (2020), Towards a better understanding of transfer learning for medical imaging, Applied Sciences 10(13), 4523.",
    ])
    DRIVE_ROOT: str = "/content/drive/MyDrive/DFU-ImageGuard"
    LOCAL_REPO: str = "/content/DFU-ImageGuard"
    RUN_ID: Optional[str] = None
    SEED: int = 2026
    N_FOLDS: int = 5
    IMAGE_SIZE: int = 224
    BATCH_SIZE: int = 24
    NUM_WORKERS: int = 2
    MAX_EPOCHS: int = 30
    PATIENCE: int = 7
    LEARNING_RATE: float = 2e-4
    WEIGHT_DECAY: float = 1e-4
    DROPOUT: float = 0.30
    TARGET_SENSITIVITY: float = 0.95
    BOOTSTRAP_REPS: int = 1000
    PHASH_DISTANCE: int = 4
    EMBEDDING_DUPLICATES: bool = True
    EMBEDDING_COSINE_THRESHOLD: float = 0.999
    FORCE_RETRAIN: bool = False
    RUN_BASELINES: bool = True
    RUN_ROBUSTNESS: bool = True
    RUN_XAI: bool = True
    XAI_CASES: int = 6
    XAI_LIME_SAMPLES: int = 150
    ROBUSTNESS_LEVELS: dict[str, list[float]] = field(default_factory=lambda: {
        "gaussian_noise": [0.03, 0.08], "gaussian_blur": [1.0, 2.5],
        "brightness": [0.70, 1.30], "contrast": [0.65, 1.35],
        "jpeg": [60, 25], "rotation": [7, 15], "occlusion": [0.10, 0.25],
    })
    PROPOSED_BACKBONE: str = "convnext_tiny"
    PRIMARY_MODEL_NAME: str = "DFU-ImageGuard"
    BASELINE_MODELS: dict[str, str] = field(default_factory=lambda: {
        "ResNet18": "resnet18", "DenseNet121": "densenet121",
        "MobileNetV3": "mobilenetv3_large_100", "EfficientNet-B0": "efficientnet_b0",
    })
    LINEAR_BASELINE_NAME: str = "Linear-LogReg"
    GITHUB_MAX_BYTES: int = 95 * 1024 * 1024

def seed_everything(seed: int) -> None:
    import torch
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

def torch_initial_seed() -> int:
    import torch
    return int(torch.initial_seed())

def _worker_init(worker_id: int) -> None:
    seed = (torch_initial_seed() + worker_id) % 2**32
    random.seed(seed); np.random.seed(seed)

def now_run_id() -> str: return dt.datetime.now().strftime("%Y%m%d_%H%M%S")

def json_default(obj: Any) -> Any:
    if isinstance(obj, Path): return str(obj)
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if dataclasses.is_dataclass(obj): return asdict(obj)
    raise TypeError(type(obj))

def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")

def read_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default

def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""): h.update(chunk)
    return h.hexdigest()

def hash_text(text: str) -> int: return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)

def relpath(path: Path, root: Path) -> str:
    try: return str(path.resolve().relative_to(root.resolve()))
    except Exception: return str(path)

class UnionFind:
    def __init__(self, n: int): self.parent=list(range(n)); self.rank=[0]*n
    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]; x=self.parent[x]
        return x
    def union(self, a: int, b: int) -> None:
        a,b=self.find(a),self.find(b)
        if a==b:return
        if self.rank[a] < self.rank[b]: a,b=b,a
        self.parent[b]=a
        if self.rank[a]==self.rank[b]: self.rank[a]+=1

def prepare_run_dirs(cfg: Config) -> dict[str, Path]:
    cfg.RUN_ID = cfg.RUN_ID or now_run_id(); root=Path(cfg.DRIVE_ROOT)/"runs"/cfg.RUN_ID
    dirs={"root":root}
    for name in ["tables","figures","models","xai","predictions","logs","configs","manifests","cache"]:
        dirs[name]=root/name; dirs[name].mkdir(parents=True, exist_ok=True)
    write_json(dirs["configs"]/"resolved_config.json", asdict(cfg)); return dirs

def mount_drive() -> None:
    try:
        from google.colab import drive
        if not Path("/content/drive/MyDrive").exists(): drive.mount("/content/drive")
    except Exception as exc: print(f"Drive mount skipped: {exc}")

def download_dataset(cfg: Config, dirs: dict[str, Path]) -> Path:
    import kagglehub
    path=Path(kagglehub.dataset_download(cfg.DATASET_HANDLE))
    if not path.exists(): raise FileNotFoundError(path)
    write_json(dirs["manifests"]/"dataset_download.json", {"handle":cfg.DATASET_HANDLE,"resolved_path":str(path),"downloaded_at":dt.datetime.now().isoformat(),"license":cfg.DATASET_LICENSE,"citations":cfg.DATASET_CITATIONS})
    return path

def normalize_folder_name(name: str) -> str: return "".join(c.lower() for c in name if c.isalnum())

def locate_strict_patch_root(dataset_root: Path) -> tuple[Path, dict[str,int]]:
    normal={"normal","normalhealthyskin","healthy","healthyskin","normalimages"}
    ulcer={"abnormal","abnormalulcer","ulcer","dfu","ulcerimages"}; candidates=[]
    for p in dataset_root.rglob("*"):
        if not p.is_dir() or normalize_folder_name(p.name)!="patches": continue
        mapping={}
        for child in p.iterdir():
            if not child.is_dir(): continue
            n=normalize_folder_name(child.name)
            if n in normal: mapping[child.name]=0
            elif n in ulcer: mapping[child.name]=1
        if set(mapping.values())=={0,1}: candidates.append((p,mapping))
    if len(candidates)!=1: raise RuntimeError(f"Expected exactly one explicit Patches class layout; found {candidates}. No labels are guessed.")
    return candidates[0]

def _pixel_hash(img: Image.Image) -> str:
    arr=np.asarray(img.convert("RGB"),dtype=np.uint8); h=hashlib.sha256(); h.update(np.asarray(arr.shape,dtype=np.int32).tobytes()); h.update(arr.tobytes()); return h.hexdigest()

def build_manifest(dataset_root: Path, cfg: Config, dirs: dict[str, Path]) -> pd.DataFrame:
    import imagehash
    patch_root,mapping=locate_strict_patch_root(dataset_root); rows=[]
    exts={".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}
    for folder,label in mapping.items():
        for path in sorted((patch_root/folder).rglob("*")):
            if not path.is_file() or path.suffix.lower() not in exts: continue
            row={"image_id":hashlib.sha256(str(path.relative_to(dataset_root)).encode()).hexdigest()[:20],"image_path":str(path),"relative_path":str(path.relative_to(dataset_root)),"source_subset":"Patches","source_class_folder":folder,"label":int(label),"label_name":"DFU" if label else "Normal","patient_id":None,"case_id":None,"exclude":False,"exclusion_reason":""}
            try:
                fh=sha256_file(path)
                with Image.open(path) as im: im.verify()
                with Image.open(path) as im:
                    rgb=im.convert("RGB"); row.update(width=rgb.width,height=rgb.height,file_sha256=fh,pixel_sha256=_pixel_hash(rgb),phash=str(imagehash.phash(rgb,hash_size=16)))
            except Exception as exc:
                row.update(exclude=True,exclusion_reason=f"corrupt_or_unreadable:{type(exc).__name__}",width=None,height=None,file_sha256=None,pixel_sha256=None,phash=None)
            rows.append(row)
    df=pd.DataFrame(rows)
    if df.empty: raise RuntimeError("No explicitly labelled Patches images found")
    df.to_csv(dirs["manifests"]/"dfu_master_manifest_initial.csv",index=False)
    write_json(dirs["manifests"]/"strict_label_mapping.json",{"patch_root":str(patch_root),"class_folder_map":mapping,"policy":"Only explicit immediate class folders under Patches are accepted."})
    return df

def _hamming_hex(a: str,b: str)->int:return (int(a,16)^int(b,16)).bit_count()

def compute_embedding_candidates(df: pd.DataFrame,cfg: Config,dirs: dict[str,Path])->list[tuple[int,int,float]]:
    if not cfg.EMBEDDING_DUPLICATES:return []
    import torch
    from torch.utils.data import DataLoader,Dataset
    from torchvision import models,transforms
    from sklearn.neighbors import NearestNeighbors
    class D(Dataset):
        def __init__(self,paths):self.paths=paths;self.tf=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
        def __len__(self):return len(self.paths)
        def __getitem__(self,i):
            with Image.open(self.paths[i]) as im:return self.tf(im.convert("RGB")),i
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model=models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1);model.fc=torch.nn.Identity();model.eval().to(device)
    loader=DataLoader(D(df.image_path.tolist()),batch_size=64,shuffle=False,num_workers=cfg.NUM_WORKERS);emb=np.zeros((len(df),512),np.float32)
    with torch.inference_mode():
        for xb,idx in loader:
            z=model(xb.to(device)).cpu().numpy();z/=np.linalg.norm(z,axis=1,keepdims=True).clip(min=1e-8);emb[idx.numpy()]=z
    np.save(dirs["cache"]/"duplicate_embeddings.npy",emb); nn=NearestNeighbors(n_neighbors=min(6,len(df)),metric="cosine").fit(emb);dist,idxs=nn.kneighbors(emb);pairs=[]
    for i in range(len(df)):
        for d,j in zip(dist[i,1:],idxs[i,1:]):
            sim=1-float(d)
            if j>i and sim>=cfg.EMBEDDING_COSINE_THRESHOLD and _hamming_hex(df.iloc[i].phash,df.iloc[j].phash)<=max(8,cfg.PHASH_DISTANCE):pairs.append((i,int(j),sim))
    pd.DataFrame(pairs,columns=["i","j","cosine_similarity"]).to_csv(dirs["tables"]/"embedding_duplicate_candidates.csv",index=False)
    del model,loader,emb;gc.collect();
    if torch.cuda.is_available():torch.cuda.empty_cache()
    return pairs

def assign_duplicate_groups(df: pd.DataFrame,cfg: Config,dirs: dict[str,Path])->pd.DataFrame:
    work=df.loc[~df.exclude].reset_index(drop=True).copy();uf=UnionFind(len(work));reasons=[]
    for col,reason in [("file_sha256","exact_file"),("pixel_sha256","exact_pixel")]:
        for idxs in work.groupby(col).groups.values():
            idxs=list(idxs)
            for j in idxs[1:]:uf.union(idxs[0],j);reasons.append({"i":idxs[0],"j":j,"reason":reason,"score":1.0})
    ph=work.phash.tolist()
    for i in range(len(work)):
        for j in range(i+1,len(work)):
            d=_hamming_hex(ph[i],ph[j])
            if d<=cfg.PHASH_DISTANCE:uf.union(i,j);reasons.append({"i":i,"j":j,"reason":"phash","score":d})
    for i,j,sim in compute_embedding_candidates(work,cfg,dirs):uf.union(i,j);reasons.append({"i":i,"j":j,"reason":"embedding_plus_phash","score":sim})
    roots=[uf.find(i) for i in range(len(work))];names={r:f"DG{n:05d}" for n,r in enumerate(sorted(set(roots)),1)};work["group_id"]=[names[r] for r in roots]
    clusters=work.groupby("group_id").agg(n_images=("image_id","size"),n_labels=("label","nunique"),labels=("label_name",lambda s:"|".join(sorted(set(s)))),member_ids=("image_id",lambda s:"|".join(s))).reset_index()
    conflicts=set(clusters.loc[clusters.n_labels>1,"group_id"]);work["label_conflict"]=work.group_id.isin(conflicts);work.loc[work.label_conflict,"exclude"]=True;work.loc[work.label_conflict,"exclusion_reason"]="duplicate_cluster_label_conflict"
    pairs=pd.DataFrame(reasons);pairs.to_csv(dirs["tables"]/"duplicate_pairs.csv",index=False);clusters.to_csv(dirs["tables"]/"duplicate_clusters.csv",index=False);work.to_csv(dirs["manifests"]/"cleaned_manifest_with_exclusions.csv",index=False)
    pd.concat([df.loc[df.exclude],work.loc[work.exclude]],ignore_index=True,sort=False).to_csv(dirs["manifests"]/"excluded_images.csv",index=False)
    cleaned=work.loc[~work.exclude].reset_index(drop=True)
    if cleaned.label.nunique()!=2:raise RuntimeError("Cleaning removed a class")
    return cleaned

def make_outer_folds(df: pd.DataFrame,cfg: Config,dirs: dict[str,Path])->pd.DataFrame:
    from sklearn.model_selection import StratifiedGroupKFold
    df=df.copy();df["outer_fold"]=-1;sg=StratifiedGroupKFold(n_splits=cfg.N_FOLDS,shuffle=True,random_state=cfg.SEED)
    for fold,(_,te) in enumerate(sg.split(df,df.label,groups=df.group_id)):df.loc[te,"outer_fold"]=fold
    integrity={"valid":True,"patient_ids_available":False,"split_unit":"duplicate_group","folds":[]}
    for fold in range(cfg.N_FOLDS):
        tr=df[df.outer_fold!=fold];te=df[df.outer_fold==fold];overlap=sorted(set(tr.group_id)&set(te.group_id));integrity["folds"].append({"fold":fold+1,"train_n":len(tr),"test_n":len(te),"train_groups":tr.group_id.nunique(),"test_groups":te.group_id.nunique(),"train_normal":int((tr.label==0).sum()),"train_dfu":int((tr.label==1).sum()),"test_normal":int((te.label==0).sum()),"test_dfu":int((te.label==1).sum()),"group_overlap":overlap});integrity["valid"]&=not overlap
    if not integrity["valid"]:raise AssertionError("Duplicate-group leakage")
    df.to_csv(dirs["manifests"]/"locked_outer_fold_assignments.csv",index=False);write_json(dirs["root"]/"split_integrity_report.json",integrity);pd.DataFrame(integrity["folds"]).to_csv(dirs["tables"]/"fold_distribution.csv",index=False);return df

def make_inner_partition(outer_train: pd.DataFrame,cfg: Config,fold: int)->pd.DataFrame:
    from sklearn.model_selection import StratifiedGroupKFold
    x=outer_train.copy().reset_index(drop=True);x["inner_fold"]=-1;sg=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=cfg.SEED+100+fold)
    for f,(_,idx) in enumerate(sg.split(x,x.label,groups=x.group_id)):x.loc[idx,"inner_fold"]=f
    x["inner_role"]="train";x.loc[x.inner_fold==fold%5,"inner_role"]="selection";x.loc[x.inner_fold==(fold+1)%5,"inner_role"]="calibration"
    roles={r:set(x.loc[x.inner_role==r,"group_id"]) for r in ["train","selection","calibration"]}
    assert not roles["train"]&roles["selection"] and not roles["train"]&roles["calibration"] and not roles["selection"]&roles["calibration"]
    for r in roles:
        if x.loc[x.inner_role==r,"label"].nunique()!=2:raise RuntimeError(f"Inner {r} lacks a class")
    return x
