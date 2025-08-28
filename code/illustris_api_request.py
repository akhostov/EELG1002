import requests
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
import numpy as np
import pickle
import pandas as pd
from tqdm import trange

baseUrl = 'http://www.tng-project.org/api/'
headers = {"api-key":"993aabb363536300837b5830bda7e65d"}

def get(path, params=None):
    """
    Sends an HTTP GET request to Illustris API with optional parameters and returns the response.

    Args:
        path (str): The URL path to send the request to.
        params (dict, optional): The parameters to include in the request. Defaults to None.

    Returns:
        Union[Response, str, dict]: The response object if the content-type is 'application/json',
            the filename string if the content-disposition header is present, or the response object itself.

    Raises:
        requests.exceptions.HTTPError: If the response code is not HTTP SUCCESS (200).
    """
    # make HTTP GET request to path
    headers = {"api-key":"993aabb363536300837b5830bda7e65d"}
    r = requests.get(path, params=params, headers=headers)
    
    # raise exception if response code is not HTTP SUCCESS (200)
    r.raise_for_status()
    
    if r.headers['content-type'] == 'application/json':
        return r.json() # parse json responses automatically
    
    if 'content-disposition' in r.headers:
        filename = r.headers['content-disposition'].split("filename=")[1]
        with open(filename, 'wb') as f:
            f.write(r.content)
        return filename # return the filename string
    
    return r


def main(simulation):
    """
        Retrieves subhalo information from the Illustris API for a given simulation and outputs the information in a pickle file.
        
        Args:
            simulation (str): The name of the simulation to retrieve data from.
            
        Returns:
            None
            
        Raises:
            None
        
        This function retrieves subhalo information from the Illustris API for a given simulation. 
        It first makes a request to the base URL to establish a connection. 
        Then, it constructs a search query to filter the subhalos based on certain criteria such as stellar mass range, minimum SFR, and ugriz photometry. 
        The search query is appended to the URL and a request is made to retrieve the subhalos. 
        The function then iterates through each subhalo and checks if it has a progenitor. 
        If it does, the subhalo ID and other relevant information are stored in a dictionary. 
        The function continues to follow the sublink URLs to retrieve the full subhalo details of the progenitor and descendant. 
        Finally, the function saves the retrieved information in a pickle file.
    """

    # Read in the Snapshot Redshift Mapping




    r = get(baseUrl)

    # Stellar Mass 
    mass_min = 10**7/  1e10 * 0.704
    mass_max = 10**(8.7) /  1e10 * 0.704
    sfr_min = 3.

    # Search Query (get the first 1000 elements)
    # Search is limited to those within a given stellar mass range, above a minimum SFR,
    # with ugriz photometry between -20.5 and -18.5
    search_query = f"?mass_stars__gt={mass_min}&mass_stars__lt={mass_max}&sfr__gt={sfr_min}&limit=1000"+\
                    f"&stellarphotometrics_u__gt=-20.5&stellarphotometrics_u__lt=-18.5"+\
                    f"&stellarphotometrics_g__gt=-20.5&stellarphotometrics_g__lt=-18.5"+\
                    f"&stellarphotometrics_r__gt=-20.5&stellarphotometrics_r__lt=-18.5"+\
                    f"&stellarphotometrics_i__gt=-20.5&stellarphotometrics_i__lt=-18.5"+\
                    f"&stellarphotometrics_z__gt=-20.5&stellarphotometrics_z__lt=-18.5"

    # form url and make request
    url = f"http://www.tng-project.org/api/{simulation}/snapshots/55/subhalos/" + search_query
    subhalos = get(url)
    print(f"Found a total of: {subhalos['count']} potential candidates")

    # Run through each and keep those that have a progenitor
    if subhalos['count'] != 0:
        for ii in trange(subhalos['count'], desc="Going through Subhalos"):

            # prepare dict to hold result arrays
            fields = ['snap','redshift','id','sfr','mass_gas','mass_stars','mass_dm','mass_bhs','halfmassrad_stars','gasmetallicity','bhmdot','starmetallicity','windmass']
            out = {}
            for field in fields:
                out[field] = []

            # Get the Initial Snap = 55 source
            subhalo_main = get(subhalos['results'][ii]['url'])

            #store the id
            id_main = subhalo_main['id']

            ssfr = np.log10(subhalo_main['sfr']/subhalo_main['mass_stars'])-10.

            if subhalo_main['related']['sublink_progenitor'] != None:
                #print(subhalos["results"][ii]['id'], ssfr, subhalo["related"]["sublink_progenitor"])

                subhalo = subhalo_main
                ind = 0 # Dummy variable just to make sure the same snapshot is not duplicated (in this case snapshot 55)
                while subhalo['prog_sfid'] != -1:
                    if ind == 0:
                        # Dummy variable just to save all subsequent snapshots but ignore 55
                        ind = 1
        
                    for field in fields:
                        if field == "redshift":
                            out[field].append(get(f"https://www.tng-project.org/api/{simulation}/snapshots/{subhalo['snap']}")["redshift"])
                        else:
                            out[field].append(subhalo[field])
                    # request the full subhalo details of the descendant by following the sublink URL
                    subhalo = get(subhalo['related']['sublink_progenitor'])

        
                subhalo = subhalo_main
                ind = 0
                while subhalo['desc_sfid'] != -1:
                    if ind == 0:
                        # Dummy variable just to save all subsequent snapshots but ignore 55
                        ind = 1

                    if ind != 0:
                        for field in fields:
                            if field == "redshift":
                                try:
                                    out[field].append(get(f"https://www.tng-project.org/api/{simulation}/snapshots/{subhalo['snap']}")["redshift"])
                                except:
                                    import pdb; pdb.set_trace()
                            else:
                                out[field].append(subhalo[field])
                    # request the full subhalo details of the descendant by following the sublink URL
                    subhalo = get(subhalo['related']['sublink_descendant'])

                # We need to sort everything
                ind = np.argsort(out["snap"])

                # Convert in to numpy arrays
                for field in fields:
                    out[field] = np.array(out[field])[ind]

                # Save all the information
                with open(f"../data/Illustris_Analogs/{simulation}_{id_main}_analogs_with_histories.pkl","wb") as outfile:
                    pickle.dump((out), outfile)

                # Close the File
                outfile.close()

    else:
        print("No subhalos found")
        exit()



if __name__ == "__main__":
    main("TNG300-2")
    main("TNG300-1")
