import pysal as ps 

def adjshapefile(file, indx):

	w = ps.queen_from_shapefile(file, indx)

	return w.full()