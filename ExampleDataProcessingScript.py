#************************************** SUPPORTING LIBRARIES
import os;
import sys;
import numpy as np;
import random;
import trimesh;

#************************************** SUPPORTING BUFFERS
mainFolder = ".";
dataFolder = mainFolder + "/Data";
headSkullShapeFolder = dataFolder + "/HeadAndSkullShapes";
crossValidationFolder = dataFolder + "/CrossValidation"
subjectIDFilePath = dataFolder + "/PostProcessedSubjectIDs_PrevPapers.txt";

#************************************** SUPPORTING FUNCTIONS
def read_strings_from_file(filename):
    """
    Reads a list of strings from a text file.
    Each line in the file is treated as a separate string.

    Args:
        filename (str): Path to the text file.

    Returns:
        List[str]: List of strings from the file.
    """
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            lines = [line.strip() for line in file if line.strip()]
        return lines
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []
def split_subject_ids(subject_ids, train_ratio=0.8, seed=42):
    """
    Splits a list of subject IDs into non-overlapping training and testing sets.

    Args:
        subject_ids (list): List of subject ID strings.
        train_ratio (float): Proportion of IDs to include in training set.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: (training_ids, testing_ids)
    """
    if seed is not None:
        random.seed(seed)

    total_ids = len(subject_ids)
    shuffled = subject_ids.copy()
    random.shuffle(shuffled)

    train_size = int(train_ratio * total_ids)

    training_ids = shuffled[:train_size]
    testing_ids = shuffled[train_size:]

    return training_ids, testing_ids
def save_strings_to_file(strings, filename):
    """
    Saves a list of strings to a text file.
    Each string will be written on a new line.

    Args:
        strings (list): List of strings to save.
        filename (str): Path to the output text file.
    """
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            for line in strings:
                file.write(f"{line}\n")
        print(f"Successfully saved {len(strings)} lines to '{filename}'.")
    except Exception as e:
        print(f"An error occurred while saving to file: {e}")
def print_progress_bar(iteration, total, prefix='', suffix='', length=50, fill='█'):
    """
    Prints a progress bar to the terminal.

    Args:
        iteration (int): Current iteration (e.g., loop index).
        total (int): Total iterations.
        prefix (str): Prefix string (e.g., 'Progress:').
        suffix (str): Suffix string (e.g., 'Complete').
        length (int): Character length of the progress bar.
        fill (str): Bar fill character.
    """
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()
    if iteration == total:
        print()  # New line on completion
def load_mesh_from_ply(file_path):
    """
    Loads a 3D mesh from a .ply file using trimesh.

    Args:
        file_path (str): Path to the .ply file.

    Returns:
        trimesh.Trimesh: The loaded mesh object, or None if loading fails.
    """
    try:
        mesh = trimesh.load(file_path, file_type='ply')
        if not isinstance(mesh, trimesh.Trimesh):
            print("Warning: Loaded object is not a single mesh.")
        return mesh
    except Exception as e:
        print(f"Failed to load mesh: {e}")
        return None
    
#************************************** PROCESSING FUNCTIONS
def trainTestSpliting():
    # Initialize
    print("Initializing ...");

    # Reading the whole subject IDs
    print("Reading the whole subject IDs ...");
    subjectIDs = read_strings_from_file(subjectIDFilePath);

    # Spliting the train and test IDs
    print("Spliting the train and test IDs ...");
    trainingIDs, testingIDs = split_subject_ids(subjectIDs);

    # Save training and testing IDs to file
    print("Saving training and testing IDs to file ...");
    trainingIDFilePath = crossValidationFolder + "/TrainingIDs.txt";
    testingIDFilePath = crossValidationFolder + "/TestingIDs.txt";
    save_strings_to_file(trainingIDs, trainingIDFilePath);
    save_strings_to_file(testingIDs, testingIDFilePath);

    # Finished processing
    print("Finished processing.");
def generateTrainingDataAndTestingData():
    # Initializing
    print("Initializing ...");

    # Forming the training data
    print("Forming the training data ...");
    ## Reading the training subject IDs
    print("\t Reading the training subject IDs ...");
    trainingIDs = read_strings_from_file(crossValidationFolder + "/TrainingIDs.txt");
    numOfTrainings = len(trainingIDs);
    ## Reading all meshes from file
    print("\tReading all training meshes from files ...");
    trainingXs = []; trainingYs = [];
    for i in range(numOfTrainings):
        # Print progress bar
        print_progress_bar(i, numOfTrainings, "\t");
        subjectID = trainingIDs[i];
    
        # Reading mesh from file
        headMesh = load_mesh_from_ply(headSkullShapeFolder + f"/{subjectID}-FullHead.ply");
        skullMesh = load_mesh_from_ply(headSkullShapeFolder + f"/{subjectID}-FullSkull.ply");
    
        # Get head vertices and skull vertices
        headVertices = headMesh.vertices;
        skullVertices = skullMesh.vertices;
    
        # Flatten head and skull vertices
        headFlattenVertices = headVertices.flatten();
        skullFlattenVertices = skullVertices.flatten();
    
        # Forming the training X and Y
        trainingXs.append(headFlattenVertices);
        trainingYs.append(skullFlattenVertices);
    trainingXs = np.array(trainingXs);
    trainingYs = np.array(trainingYs);

    # Forming the testing data
    print("Forming the testing data ...");
    ## Reading the testing subject IDs
    print("\t Reading the testing subject IDs ...");
    testingIDs = read_strings_from_file(crossValidationFolder + "/TestingIDs.txt");
    numOfTestings = len(testingIDs);
    ## Reading all meshes from file
    print("\t Reading all testing meshes from files ...");
    testingXs = []; testingYs = [];
    for i in range(numOfTestings):
        # Print progress bar
        print_progress_bar(i, numOfTestings, "\t ");
        subjectID = testingIDs[i];
    
        # Reading mesh from file
        headMesh = load_mesh_from_ply(headSkullShapeFolder + f"/{subjectID}-FullHead.ply");
        skullMesh = load_mesh_from_ply(headSkullShapeFolder + f"/{subjectID}-FullSkull.ply");
    
        # Get head vertices and skull vertices
        headVertices = headMesh.vertices;
        skullVertices = skullMesh.vertices;
    
        # Flatten head and skull vertices
        headFlattenVertices = headVertices.flatten();
        skullFlattenVertices = skullVertices.flatten();
    
        # Forming the testing X and Y
        testingXs.append(headFlattenVertices);
        testingYs.append(skullFlattenVertices);
    testingXs = np.array(testingXs);
    testingYs = np.array(testingYs);

    # Finished processing.
    print("Finished processing.");

#************************************** MAIN FUNCTION
def main():
    os.system("cls");
    generateTrainingDataAndTestingData();
if __name__ == "__main__":
    main();