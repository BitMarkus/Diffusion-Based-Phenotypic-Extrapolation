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

        if setting.get('ds_save_val_images', False):
            print("Exporting validation images. Please wait....")
            val_img_export_pth = setting['pth_ds_gen_output'] / "validation_images"
            if not val_img_export_pth.exists():
                val_img_export_pth.mkdir(parents=True, exist_ok=True)
            self.ds.export_validation_images(val_img_export_pth)

        if self.ds.ds_loaded:
            print("Training and validation datasets successfully loaded.")
            print(f"Number training images/batches: {self.ds.num_train_img}/{self.ds.num_train_batches}")
            print(f"Number validation images/batches: {self.ds.num_val_img}/{self.ds.num_val_batches}")

    # Train Network.
    def _train_network(self, device) -> None:
        print("\n:TRAIN NETWORK:")
        print("  Results will be saved to output/train/[timestamp]/")
        if not self.cnn_wrapper.model_loaded:
            print('No CNN generated yet!')
        elif not self.ds.ds_loaded:
            print('No training data loaded yet!')
        else:
            print("Start training...")
            train = Train(self.cnn_wrapper, self.ds, device)
            train.train()
            print("\nTraining finished!")

    # Load Weights.
    def _load_weights(self) -> None:
        print("\n:LOAD WEIGHTS:")
        if not self.cnn_wrapper.model_loaded:
            print('No CNN generated yet!')
        else:
            self.cnn_wrapper.load_checkpoint()