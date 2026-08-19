import utils.physics_engine as phys
from utils.model_loader import ModelLoader
from utils.editor import Editor
from utils.simulation import run_simulation

def main():
    print("[INFO] Starting PhysGaussian Studio (Low-Poly Mesh Mode)...")
    
    # Load initial model and data with the use_mesh flag enabled!
    loader = ModelLoader(
        max_render_points=phys.MAX_RENDER_POINTS, 
        max_phys_points=phys.MAX_PHYS_POINTS,
        use_mesh=True
    )
    loader._load_worker("modello.ply")
    init_data_dict = loader.get_loaded_data()
    
    print(f"[INFO] Physical particles: {init_data_dict['N']} | Rendered Gaussians: {init_data_dict['M']}")
    print("[OK] Skinning completed.")
    
    # Initialize and run Editor
    editor = Editor(loader, init_data_dict)
    trigger_simulation, sim_params = editor.run()

    # If the user clicked "Start Simulation and Export", run the offline simulator
    if trigger_simulation and sim_params is not None:
        run_simulation(*sim_params)
        print("\nOperation completed! Video is ready.")


if __name__ == "__main__":
    main()
