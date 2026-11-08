from script2stlite import Script2StliteConverter

# Initialize the converter with the path to the example app
# If you have cloned the repository, this path will be relative to the repo root.
converter = Script2StliteConverter(directory=".")

# Prepare the folder. This creates 'settings.yaml' if it's missing
# and ensures a 'pages' directory exists.
# For this example, 'settings.yaml' is already provided.
converter.prepare_folder()

converter.convert()
