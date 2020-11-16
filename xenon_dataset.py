from torch_geometric.data import Data, Dataset
from torch_geometric.utils import from_networkx
import networkx as nx
import pickle as pk
import numpy as np
import os

class XenonDataset(Dataset):

    def _paths_exist(self):
        """
        These paths should exist for the files. Returns True, [] if all paths exist. Returns
        False and a list of the missing paths.
        """
        all_present = True
        missing = []
        if not os.isdir( os.path.join(self.root_dir, "graph") ):
            all_present = False
            missing.append("graph")
        if not os.isdir( os.path.join(self.root_dir, "data") ):
            all_present = False
            missing.append("data")
        return all_present, missing

    def __init__(self, root_dir, data_name, graph_name, proc_name=None, training_split=0.8,
                 train=True, simulation=True, transform=None, pre_transform=None):
        self.root_dir = root_dir
        all_present, missing = self._paths_exist()
        # Need to have the data directory and the graph directory present to pull from
        assert all_present, "There are directories missing! {}".format(missing)

        self.train = train # Training or validation set
        self.training_split = training_split
        assert training_split <= 1 and training_split >= 0, \
            "The split for training/validation is not between [0,1]"

        self.data_name = data_name
        self.graph_name = graph_name
        self.simulation = simulation # Simulated or experimental data
        if proc_name == None: # Use the raw_name for the subdir name
            self.proc_name = data_name.split('.')[0] # Removing file extensions
        else:
            self.proc_name = proc_name
        self.content_size = 0 # Tracks the number of graphs

        # Check if there is already a processed dataset. If there is, read from that dataset, does
        # not overwrite
        if os.path.isdir( os.path.join(root_dir, 'processed', self.proc_name, "") ):
            self.content_size = len(os.listdir(
                os.path.join(root_dir, 'processed', self.proc_name,"") ))
        super(XenonDataset, self).__init__(root_dir, transform, pre_transform)

    @property
    def data_file_names(self):
        return self.data_name

    @property
    def processed_file_names(self):
        return self.proc_name

    def process(self):
        """
        Processes two pickle files into a series of graph files. The first pickle file is the
        simulation/experimental data that contains the light seen by the PMT and the true position
        of the event (when given a simulation). The second pickle file is the graph structure to
        use for the GCNN.
        """

        # Graph structure get
        with open( os.path.join(self.root_dir, "data", self.graph_name), 'rb' ) as fn:
            input_graph = pk.load(fn)
        torch_graph = from_networkx(input_graph)

        # Starting to process the data files.
        i = 0
        for data_path in self.raw_paths:
            with open(data_path, 'rb') as fn:
                data_contents = pk.load(fn)
            self.content_size += len(raw_contents)
            for event_data in data_contents:
                # Light detecting by each PMT
                x = event_data['area_per_channel'][:127] # Only getting the top PMTs
                x = np.reshape( x, (len(x), 1) )
                x = np.hstack( (x, np.array(torch_graph.pos)) ).astype(np.float32)

                # True position of this event
                if self.simulation:
                    y = np.reshape( event_data['true_pos'], (1, len(event_data['true_pos'])) )
                else: # Not a simulation -> no truth
                    y = None

                processed_data = Data(x=torch.tensor(x), edge_index=torch_graph.edge_index,
                                      y=torch.tensor(y), pos=torch_graph.pos)

                # Saving the contents into the processed directory
                path_to_proc = os.path.join(self.processed_dir, self.proc_anem)
                if not os.path.isdir(path_to_proc):
                    os.makedirs(path_to_proc)
                torch.save(processed_data, os.path.join(path_to_proc, 'graph_{}.pt'.format(i)) )
                i += 1

    def get(self, idx):
        """ Get the graph_$idx$.pt from $processed_dir$. """
        path_to_proc = os.path.join(self.processed_dir, self.proc_name)

        # If it's experimental data
        if not self.simulation:
            return torch.load( os.path.join(path_to_proc, 'graph_{}.pt'.format(idx)) )

        # It it's simulated data
        if self.train:
            idx_shift = 0
        else:
            idx_shift = int(self.content_size * self.train)
