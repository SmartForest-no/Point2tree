class DataQualityError(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        full_message = (
            "\n\n###########################################################################\n"
            + "DATA QUALITY ERROR: "
            + self.message
            + "\n###########################################################################"
            "\nThis has most likely failed due to poor semantic segmentation results. "
            "\nThis typically means that your input point cloud was not suitable for FSCT. "
            '\nHave a look at "segmented.las". It will likely be very poorly segmented.'
            "\nPlease try again with a higher-resolution and/or higher-quality point cloud. "
            "\n"
            '\nTry processing the file "example.las" in FSCT/data/train/ '
            "\nIf FSCT doesn't finish successfully with this file, you either have "
            "\ninstallation issues, or there is a bug in the code."
            "\n###########################################################################\n"
        )
        return full_message


class NoDataFound(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        full_message = (
            "\n\n###########################################################################\n"
            + "NO DATA FOUND ERROR: "
            + self.message
            + "\n###########################################################################"
        )
        return full_message
