# ===== Own Modules =====
import functions as fn
from settings import setting
from . import Train, CNN_Model, Dataset

class SingleTrainingMenu:

    # Single CNN Training Menu - Tools for creating, training, and managing a single CNN model.

    #############################################################################################################
    # CONSTRUCTOR

    def __init__(self) -> None:

        # Fresh state on each entry into the submenu
        self.ds = Dataset()
        self.cnn_wrapper = CNN_Model()

    #############################################################################################################
    # METHODS

    # Display the Single CNN Training submenu and route to the selected option.
    # Args:
    #   device (torch.device): Device to run operations on
    def menu(self, device) -> None:
        while True:
            print("\n:SINGLE CNN TRAINING:")
            print("1) Create CNN Network")
            print("2) Show Network Summary")
            print("3) Load Training Data")
            print("4) Train Network")
            print("5) Load Weights")
            print("6) Back to Main Menu")

            choice = fn.input_int("Please choose: ")

            if choice == 1:
                self._create_cnn_network(device)
            elif choice == 2:
                self._show_network_summary(device)
            elif choice == 3:
                self._load_training_data()
            elif choice == 4:
                self._train_network(device)
            elif choice == 5:
                self._load_weights()
            elif choice == 6:
                print("\nReturning to main menu...")
                break
            else:
                print("Not a valid option!")

    # Create CNN Network.
    def _create_cnn_network(self, device) -> None:
        print("\n:NEW CNN NETWORK:")
        if self.cnn_wrapper.model_loaded:
            print("A network was already loaded!")
        else:
            if self.cnn_wrapper:
                self.cnn_wrapper.print_class_list()
                print(f"Creating new {self.cnn_wrapper.cnn_type} network...")
                cnn = self.cnn_wrapper.load_model(device).to(device)
                print("New network was successfully created.")
                self.cnn_wrapper.print_model_size()
            else:
                print("Unable to load the requested cnn architecture!")

    # Show Network Summary.
    def _show_network_summary(self, device) -> None:
        print("\n:SHOW NETWORK SUMMARY:")
        if self.cnn_wrapper.model_loaded:
            self.cnn_wrapper.model_summary(device)
        else:
            print("No network was generated yet!")

    # Load Training Data.
    def _load_training_data(self) -> None:
        print("\n:LOAD TRAINING DATA:")
        self.ds.validate_validation_settings()
        self.ds.load_training_dataset()
        if self.ds.validation_from_test:
            self.ds.load_test_dataset()
        self.ds.print_dataset_info()

        if self.ds.ds_loaded:
            print("Training and validation datasets successfully loaded.")
            print(f"Number training images/batches: {self.ds.num_train_img}/{self.ds.num_train_batches}")
            print(f"Number validation images/batches: {self.ds.num_val_img}/{self.ds.num_val_batches}")

    # Train Network.
    # Fully automated: builds a fresh Dataset and CNN_Model internally, so the
    # user does not need to use options 1 and 3 before training.
    # If ds_shuffle_seed is a list with multiple entries, runs one training per seed,
    # with each run saved to output/train/<parent_timestamp>/seed_<NN>/.
    # For a single seed, uses the standard output/train/<timestamp>/ folder.
    def _train_network(self, device) -> None:
        from datetime import datetime

        print("\n:TRAIN NETWORK:")

        seeds = self.ds.shuffle_seed_list
        multi_seed = len(seeds) > 1

        if multi_seed:
            print(f"  Seeds: {seeds}")
            print(f"  Mode: multi-seed sweep ({len(seeds)} runs)")
            parent_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            parent_dir = setting['pth_output'] / "train" / parent_ts
            parent_dir.mkdir(parents=True, exist_ok=True)
            print(f"  Results will be saved to: {parent_dir}")
        else:
            print(f"  Seed: {seeds[0]}")
            print(f"  Mode: single training")
            print(f"  Results will be saved to: output/train/<timestamp>/")
            parent_dir = None

        for seed in seeds:
            print(f"\n{'='*60}")
            print(f">>> TRAINING WITH SEED {seed}")
            print(f"{'='*60}")

            # Fresh dataset for this seed
            ds = Dataset()
            ds.set_shuffle_seed(seed)
            ds.validate_validation_settings()
            ds.load_training_dataset()
            if ds.validation_from_test:
                ds.load_test_dataset()

            if not ds.ds_loaded:
                print(f"✗ Dataset failed to load for seed {seed}. Skipping.")
                continue

            print(f"Number training images/batches: {ds.num_train_img}/{ds.num_train_batches}")
            print(f"Number validation images/batches: {ds.num_val_img}/{ds.num_val_batches}")

            # Fresh model for this seed
            cnn_wrapper = CNN_Model()
            cnn_wrapper.load_model(device)

            # Determine output directory for this seed
            if multi_seed:
                output_dir = parent_dir / f"seed_{seed}"
            else:
                output_dir = None  # Train will generate timestamp folder

            # Train
            train = Train(cnn_wrapper, ds, device, output_dir=output_dir)
            train.train()

            print(f">>> Training with seed {seed} finished.")

        print(f"\n{'='*60}")
        print("All trainings completed.")
        print(f"{'='*60}")

    # Load Weights.
    def _load_weights(self) -> None:
        print("\n:LOAD WEIGHTS:")
        if not self.cnn_wrapper.model_loaded:
            print('No CNN generated yet!')
        else:
            self.cnn_wrapper.load_checkpoint()