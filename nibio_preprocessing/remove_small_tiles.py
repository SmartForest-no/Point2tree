import glob
import os
from numpy import number
import plyfile


class RemoveSmallTiles():
    """
    Works with ply files.
    """
    def __init__(self, dir, min_density, tile_index_file, verbose=False):
        self.dir = dir
        self.min_size = min_density
        self.tile_index_file = tile_index_file
        self.verbose = verbose
        
    def get_density_and_points_nr_of_single_tile(self, tile_path):
        """
        Get the density of a single tile.
        """
        plydata = plyfile.PlyData.read(tile_path)
        volume = (plydata['vertex']['x'].max() - plydata['vertex']['x'].min()) \
        * (plydata['vertex']['y'].max() - plydata['vertex']['y'].min()) \
        * (plydata['vertex']['z'].max() - plydata['vertex']['z'].min())

        number_of_points = plydata['vertex'].count
        density = number_of_points / volume

        return density, number_of_points

    @staticmethod
    def remove_line_from_csv(path, line):
        """
        Remove a line from a csv file.
        """
        with open(path, 'r') as f:
            lines = f.readlines()
        with open(path, 'w') as f:
            for i in range(len(lines)):
                if i != line:
                    f.write(lines[i])

    def get_density_of_all_tiles(self):
        """
        Get the density of all tiles in the directory.
        """
        files = glob.glob(os.path.join(self.dir, "*.ply"))

        if self.verbose:
            print(f'Found {len(files)} tiles')

        # compute the density of each tile and put it in a dictionary toghehter with the file path
        densities_and_point_nr = {}
        for i, ply in enumerate(files):
            # get a fine name without the path and suffix
            file_name = os.path.split(ply)[1].split('.')[0]
            # full path to the tile
            tile_path = os.path.join(self.dir, file_name + '.ply')
            # get the density of the tile
            density = self.get_density_and_points_nr_of_single_tile(ply)
            # put it into a tuple together with the file name
            densities_and_point_nr[i] = (file_name, tile_path, density[0], density[1])

        if self.verbose:
            print(f'Computed density of all tiles')

        return densities_and_point_nr


    def remove_small_tiles(self):
        """
        Remove tiles that are too small.
        """
        # get the density of all tiles
        densities_and_point_nr = self.get_density_of_all_tiles()

        # remove the tiles that whicha are too small (points per m^3)
        for i, density in densities_and_point_nr.items():
            if density[3] < 10000:
                if self.verbose:
                    print(f'Removing {density[1]} which has only {density[3]} points')
                os.remove(density[1])
                self.remove_line_from_csv(self.tile_index_file, i)

        # remove the tiles that of a small density
        for i, density in densities_and_point_nr.items():
            if density[2] < self.min_size:
                if self.verbose:
                    print(f'Removing tile {density[0]} with density {density[2]}')
                # check if the tile is already removed
                if os.path.exists(density[1]):
                    os.remove(density[1])
                    self.remove_line_from_csv(self.tile_index_file, i)

        # if verbose print the lowest and the highest density along with the tile name
        if self.verbose:
            print(f'Lowest density: {min(densities_and_point_nr.values(), key=lambda x: x[2])}')
            print(f'Highest density: {max(densities_and_point_nr.values(), key=lambda x: x[2])}')

        # if verbose print the number of tiles that were removed
        if self.verbose:
            print(f'Removed {len(densities_and_point_nr) - len(glob.glob(os.path.join(self.dir, "*.ply")))} tiles')

    def main(self):
        self.remove_small_tiles()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True, help='Path to the folder with the tiles')
    parser.add_argument('--min_density', type=float, required=True, help='Minimum density of the tiles')
    parser.add_argument('--tile_index_file', type=str, required=True, help='Path to the tile index file')
    parser.add_argument('--verbose', action='store_true', help='Print the tiles that are removed')
    args = parser.parse_args()

    remove_small_tiles = RemoveSmallTiles(
        args.dir, 
        args.min_density, 
        args.tile_index_file, 
        args.verbose)

    remove_small_tiles.main()

     
        