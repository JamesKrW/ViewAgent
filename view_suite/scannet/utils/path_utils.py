import os



def resolve_scene_ply(scannet_root: str, scene_id: str) -> str:
    """
    Given scannet_root and scene_id, resolve the absolute path to a .ply mesh.
    Prefer *_vh_clean.ply, fall back to *_vh_clean_2.ply.
    # example:
    # scannet_root = "~/scannet/scans"
    # scene_id = "scene0011_00"
    # resolve_scene_ply(scannet_root, scene_id)
    # "~/scannet/scans/scene0011_00/scene0011_00_vh_clean.ply"
    """
    scene_dir = os.path.join(scannet_root, scene_id)
    ply1 = os.path.join(scene_dir, f"{scene_id}_vh_clean.ply")    
    ply2 = os.path.join(scene_dir, f"{scene_id}_vh_clean_2.ply")    
    if os.path.exists(ply1):
        return ply1
    if os.path.exists(ply2):
        return ply2
    raise FileNotFoundError(f"No valid ply file found for scene {scene_id} under {scene_dir}")


def resolve_scene_gs_ply(gs_root: str, scene_id: str) -> str:
    """
    Given gs_root (3DGS dataset root) and scene_id, resolve the absolute
    path to the pretrained 3DGS PLY checkpoint.

    Expected layout (matches GaussianWorld/scannet_mcmc_1.5M_3dgs):
      gs_root/scene_id/ckpts/point_cloud_30000.ply
    """
    path = os.path.join(gs_root, scene_id, "ckpts", "point_cloud_30000.ply")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No 3DGS PLY found for scene {scene_id} at {path}. "
            f"Did you run scripts/scannet/download_scannet_service_3dgs.sh?"
        )
    return path