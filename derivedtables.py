import abc
from datetime import datetime
from config import SimConfig
from simlog import SimLog
import simulation
import snapshot

from tqdm import tqdm

import galaxy

import utils
from utils import *

class ValidateRuleExisted():
    '''
    Validate whether a table has been created with a same set of parameters.
    '''

    def __init__(self, **params_to_validate):
        self.params = params_to_validate

    def validate(self, event, verbose="normal"):
        # Do all the current parameters match those in the log file?
        for key in self.params:
            if(self.params[key] != event.get(key)):
                talk("Log has {} = {} different from {}".format(
                    key, event.get(key), self.params[key]), "normal", verbose)
                return False

        # Does the table exist?
        file_type = event.get('file_type')
        if(file_type is None):
            file_type == "file"
        if(file_type == 'file' and (not os.path.exists(event.get('path')))):
            return False
        elif(file_type == 'dir' and (not os.path.isdir(event.get('path')))):
            return False

        return True

class ValidateRuleWithField():
    '''
    Validate whether a table has a particular field (column).
    '''
    def __init__(self, fields, columns):
        self.fields = fields
        self.columns = columns

    def validate(self, event, verbose="normal"):
        fields = self.fields
        if(isinstance(fields, str)):
            fields = [fields]

        for field in fields:
            record = self.event.get("with_field_" + field.lower())
            if(record is None or not record):
                talk("{} field is not found.".format(field), "normal", self.verbose)
                return False
            if(field not in columns):
                talk("{} field is not found in the columns of the current table.", "normal", self.verbose)

        return True

class DerivedTable(abc.ABC):
    '''
    Interface for derived tables.
    '''
    
    def __init__(self, model, cfg=SimConfig()):
        self._model = model
        self._cfg = cfg
        self._log = SimLog(model)
        self.data = None
        self._path_out = ""
        self._schema = {}
        self._clsstr = ""
        self._path_table = None
        self._pars = {}

    @abc.abstractmethod
    def build_table(self):
        pass

    @abc.abstractmethod
    def load_table(self):
        pass

    @abc.abstractmethod
    def validate_table(self):
        pass

    # @abc.abstractmethod
    # def delete_table(self):
    #     pass

    @abc.abstractmethod
    def save_table(self):
        pass

    @abc.abstractmethod
    def get_path(self):
        pass

    @abc.abstractmethod
    def _update_log(self):
        pass

    def set_param(self, key, value):
        self._pars[key] = value

    @staticmethod
    def load_schema(class_string):
        schema = utils.load_default_schema(
            SimConfig().get('Schema', 'derivedtables'))
        return schema[class_string]

    @property
    def model(self):
        return self._model

    @property
    def clsstr(self):
        return self._clsstr

class TemporaryTable(DerivedTable):

    def __init__(self, model, snapnum, galIdTarget):
        super(TemporaryTable, self).__init__(model)
        self._model = model
        self._snapnum = snapnum
        self._simulation = Simulation(model)
        self._galId = galIdTarget
        self._path_out = self._simulation._path_tmpdir
        self._fformat = "parquet"
        self._clsstr = ""
        self._schema = {}

        # Set default filename
        self.filename = ""
        self._path_table = self.get_path()
        self._pars = {
            'snapnum': snapnum,
            'galId': galIdTarget,
            'fformat': self._fformat,
            'file_type': "file"
        }

    def build_table(self):
        '''
        Implement your own build_table method here.
        '''
        pass

    def _update_log(self):
        self._log.write_event(self.filename, self._pars)

    def load_table(self, validate=True, spark=None, verbose='talky'):
        '''
        Load existing (temporary) table.

        Parameters
        ----------
        validate: Boolean. Default=True
            If True, validate with the log file to verify that the saved table
            was created with the same parameters.
        spark: SparkSession. Default=None
            If None, return the table as a pandas dataframe.
            Otherwise, return the table as a Spark dataframe.
        verbose: String. Default='talky'
            The verbose level.

        Return
        ------
        Boolean.
        '''

        if(validate and self.validate_table(self.validate_rule)):
            talk("Loading {} from file.".format(self.clsstr), 'quiet')

            if(spark is None): 
                if(self._pars['fformat'].lower() == "parquet"):
                    self.data = pd.read_parquet(self._path_table)
                elif(self._pars['fformat'].lower() == "csv"):
                    self.data = pd.read_csv(self._path_table,
                                            dtype=self._schema['dtypes'])
                return True
            else:
                if(self._pars['fformat'].lower() == "parquet"):
                    self.data = spark.read.parquet(self._path_table)
                elif(self._pars['fformat'].lower() == "csv"):
                    self.data = spark.read.csv(self._path_table)
            
        else:
            talk("{} not found. Use build_table() to build new table.".format(self._path_table), verbose)
            return False

    def validate_table(self, validate_rule, verbose="normal"):
        '''
        Validate that file already exists and the parameters match the current 
        instance.
        '''

        # Is a record found in the log file?
        event = self._log.load_event(self.filename)
        if(event is None):
            return False

        return validate_rule.validate(event, verbose)

    def save_table(self, spark=None):
        '''
        TemporaryTable.save_table()
        Save the created/updated table and update log file.

        Parameters
        ----------
        spark: SparkSession. Default=None
            If None, do not use spark
        '''

        talk("Saving {} as {}".format(self.clsstr, self._path_table), 'quiet')

        if(spark is None):
            if(isinstance(self.data, pyspark.sql.dataframe.DataFrame)):
                try:
                    self.data.toPandas()
                except:
                    raise RuntimeError("Can not convert Spark DataFrame to Pandas Dataframe.")
            
            if(self._pars.get('fformat') == "parquet"):
                schema = utils.pyarrow_read_schema(self._schema)
                tab = pa.Table.from_pandas(self.data, schema=schema,
                                           preserve_index=True)
                pq.write_table(tab, self._path_table)
            elif(self._pars.get('fformat') == "csv"):
                self.data.reset_index().to_csv(self._path_table, index=False,
                                               columns=self._schema['columns'])
            self.set_param('file_type', 'file')
        else: # use Spark
            if(isinstance(self.data, pd.DataFrame)):
                try: 
                    self.data = spark.createDataFrame(self.data)
                except:
                    warnings.warn("Can not convert Pandas DataFrame to Spark Dataframe. Write without spark instead.")
                self.save_table(spark=None)
                
            if(self._pars.get('fformat') == "parquet"):
                self.data.write.mode("overwrite").parquet(self._path_table)
            elif(self._pars.get('fformat') == "csv"):
                self.data.write.mode("overwrite").csv(self._path_table)
            
            self.set_param('file_type', 'dir')
            
        self._update_log()

    def data(self):
        return self.data

    def get_path(self):
        '''     
        Get the full path to the table.
        '''
        fname = self.filename
        path_table = os.path.join(self._path_out, fname)
        return path_table

    def _initialize(self):
        self.filename = ""
        self._schema = self.load_schema(self._clsstr)
        self._path_table = self.get_path()
        self._pars['path'] = self._path_table

    @property
    def filename(self):
        return self._filename

    @filename.setter
    def filename(self, name):
        if(name == ""):
            name = "{}_{:03d}_{:05d}.{}".format(
                self.clsstr, self.snapnum, self._galId, self._fformat)
        self._filename = name

    @property
    def snapnum(self):
        return self._snapnum

    @property
    def galId(self):
        return self._galId


class PermanentTable(DerivedTable):

    def __init__(self, model):
        super(PermanentTable, self).__init__(model)        
        self._model = model
        self._simulation = simulation.Simulation(model)
        self._path_out = self._simulation._path_workdir
        self._fformat = "csv"
        self._clsstr = ""
        self._schema = {}

        # Set default filename
        self.filename = None
        self._path_table = None
        self._pars = {
            'fformat': self._fformat,
            'file_type': "file"
        }

    def _parse_keywords(self, **kwargs):
        '''
        Parse keywords specific to PermanentTable
        '''
        if(kwargs.get('fformat') is not None):
            self._pars['fformat'] = kwargs['fformat']
        else:
            self._pars['fformat'] = 'csv'

    def build_table(self):
        '''
        Implement your own build_table method here.
        '''
        pass

    def _update_log(self):
        self._log.write_event(self.filename, self._pars)

    def load_table(self, validate=True, verbose='talky'):
        '''
        Load existing table.
        '''

        if(validate and self.validate_table(self.validate_rule)):
            talk("Loading {} from file.".format(self.clsstr), 'quiet')
            if(self._pars['fformat'].lower()) == "parquet":
                self.data = pd.read_parquet(self._path_table)
            elif(self._pars['fformat'].lower()) == "csv":
                self.data = pd.read_csv(self._path_table,
                                        dtype=self._schema['dtypes'])
            return True
        else:
            talk("{} not found. Use build_table() to build new table.".format(self._path_table), verbose)
            return False

    def validate_table(self, validate_rule, verbose="normal"):
        '''
        Validate that file already exists and the parameters match the current 
        instance.
        '''

        # Is a record found in the log file?
        event = self._log.load_event(self.filename)
        if(event is None):
            return False

        return validate_rule.validate(event, verbose)

    def save_table(self, spark=None):
        '''
        PermanentTable.save_table()
        Save the created/updated table and update log file.

        Parameters
        ----------
        spark: SparkSession. Default=None
            if None, do not use spark.
        '''

        talk("Saving {} as {}".format(self.clsstr, self._path_table), 'quiet')

        if(self._pars.get('fformat') == "parquet"):
            if(spark is None):
                schema_arrow = utils.pyarrow_read_schema(self._schema)
                tab = pa.Table.from_pandas(self.data.reset_index(),
                                           schema=schema_arrow,
                                           preserve_index=False)
                pq.write_table(tab, self._path_table)
            else:
                self.data.write.mode('overwrite').parquet(self._path_table)
                
        elif(self._pars.get('fformat') == "csv"):
            if(spark is None):
                self.data.reset_index().to_csv(self._path_table, index=False,
                                               columns=self._schema['columns'])
            else:
                self.data.write.mode('overwrite').csv(self._path_table)
                
        else:
            raise ValueError("save_table() supports only csv and parquet format.")

        if(spark is not None):
            self.set_param('file_type', 'dir')

        self._update_log()

    def get_path(self):
        '''     
        Get the full path to the table.
        '''
        fname = self.filename
        path_table = os.path.join(self._path_out, fname)
        return path_table

    def _initialize(self):
        self.filename = ""
        self._schema = self.load_schema(self._clsstr)
        self._path_table = self.get_path()
        self._pars['path'] = self._path_table

    @property
    def filename(self):
        return self._filename

    @filename.setter
    def filename(self, name):
        if(name == ""):
            name = "{}.{}".format(self.clsstr, self._fformat)
        self._filename = name

# ----------------------------------------------------------------
#                      Class: GasPartTable
# ----------------------------------------------------------------

class GasPartTable(TemporaryTable):
    '''
    Temporary table that contains selected gas particles and their histories.
    '''
    
    def __init__(self, model, snapnum, galIdTarget, **kwargs):
        super(GasPartTable, self).__init__(model, snapnum, galIdTarget)
        self._clsstr = "gptable"
        self._initialize()
        self._parse_keywords(**kwargs)

        # Must match the following parameters to validate an existinig table
        self.validate_rule = ValidateRuleExisted(
            snapnum=self._snapnum,
            galId=self._galId,
            include_stars=self._pars.get('include_stars'),
            fformat=self._pars.get('fformat')
        )
        
    def _parse_keywords(self, **kwargs):
        '''
        Parse keywords specific to GasPartTable.
        '''
        
        if(kwargs.get('include_stars') is not None):
            self._pars['include_stars'] = kwargs['include_stars']
        if(kwargs.get('with_field_mgain') is not None):
            self._pars['with_field_mgain'] = kwargs['with_field_mgain']
        if(kwargs.get('with_field_relation') is not None):
            self._pars['with_field_relation'] = kwargs['with_field_relation']

    def build_table(self, pidlist, splittable, include_stars=False, overwrite=False):
        '''
        Build the gptable for the particles to track. The particles are 
        specified by their particle IDs in the pidlist.
        
        Parameters
        ----------
        pidlist: list.
            List of particle IDs for which to build the table
        splittable: pandas.DataFrame
            Table that contains information for all particle split events
        include_stars: boolean. Default = False.
            If True, track the history star particles along with gas particles.
        overwrite: boolean. Default=False
            If True, rebuild the table even if a file exists.
        '''
        
        self.set_param('include_stars', include_stars)
        
        snaplast = self.snapnum

        if(overwrite==False and self.validate_table(self.validate_rule)):
            talk("Load existing {}.".format(self.clsstr), 'normal')
            self.load_table()
            return

        talk("Building {} for {} gas particles".format(self.clsstr, len(pidlist)), 'normal')

        # First find all the ancestors
        talk("Finding ancestors for gas particles.", "normal")
        ancestors = self.find_particle_ancestors(pidlist, splittable)
        
        gptable = None
        for snapnum in tqdm(range(0, snaplast+1), desc='snapnum', ascii=True):
            snap = snapshot.Snapshot(self.model, self.snapnum, verbose='talky')
            snap.load_gas_particles(['PId','Mass','haloId'])

            # From pidlist infer all PIds needed from this snapshot
            # From the ancestor table, find the smallest snapnext that is larger than the current snapnum. For example, the history of particle Id=8 being:
            # PId   4   4   6   6   6   6   6   6   6   8   8   8
            # Gen   2   2   1   1   1   1   1   1   1   0   0   0
            # Mass  m/4 m/4 m/2 m/2 m/2 m/2 m/2 m/2 m/2 m   m   m
            #               s1              cur         s2
            # At current time, we found the next splitting at s2, where:
            # PId=8, parentId=6, snapnext=s2

            # TODO: Build the ancestors table
            parents = ancestors.query('snapnext > @snapnum').reset_index()
            parents = parents.loc[parents.groupby('PId').gen.idxmax()].reset_index()
            pidtosearch = set(pidlist + list(parents.parentId.drop_duplicates()))
            
            gp = snap.gp.loc[snap.gp.PId.isin(pidtosearch), :]
            gp.loc[:,'snapnum'] = snapnum

            # Replace the ancestors' PIds at the current snapshot with the PIds of the descendents (when there are multiple descendents, make a copy for each descendent, INCLUDING itself (PId == parentId, but the particle has splitted)).
            # Might be multiple instances parentId -> multiple PId
            gp_parents = pd.merge(gp, parents[['PId', 'parentId', 'gen']],
                                  left_on='PId', right_on='parentId',
                                  suffixes=("_l", None))
            # Should reduce particle Mass according to its generation
            gp_parents.Mass = gp_parents.Mass / (2.0 ** gp_parents.gen)
            # gp_parents.drop(['PId_l', 'parentId', 'gen'], axis=1, inplace=True)
            gp_parents.drop(['PId_l', 'parentId'], axis=1, inplace=True)
            gp = gp[gp.PId.isin(pidlist)]
            gp = gp[~gp.PId.isin(gp_parents.PId)]

            if(include_stars):
                snap.load_star_particles(['PId','Mass','haloId'])
                # Early time, no star particle
                if(snap.sp is not None):
                    sp = snap.sp.loc[snap.sp.PId.isin(pidlist), :]
                    if(not sp.empty):
                        sp.loc[:,'snapnum'] = snapnum
                        gp = pd.concat([gp, sp])

            # The star particles and 'virgin' gas particle has gen=0.
            gp['gen'] = 0

            # Add those particles that have been a parent
            gp = pd.concat([gp, gp_parents])

            # Attach to the main table
            gptable = gp.copy() if gptable is None else pd.concat([gptable, gp])

        gptable = gptable.set_index('PId').sort_values('snapnum')
        self.data = gptable

        # New table is clean without the derived fields
        self.set_params('with_field_mgain', False)
        self.set_params('with_field_relation', False)

    def add_field_mgain(spark=None):
        '''
        For each gas particle in the gptable, compute its mass gain since last
        snapshot. This does not account for splitting particles yet. 
        '''

        if(self.validate_table(ValidateRuleWithField('Mgain', self.data.columns))):
            return
        
        if(spark is None):
            # This is a very expensive operation
            self.data = self.data.reset_index()
            self.data['Mgain'] = self.data.groupby(self.data.PId).Mass.diff().fillna(0.0)
        else:
            w = Window.partitionBy(self.data.PId).orderBy(self.data.snapnum)
            self.data.withColumn('Mgain', self.data.Mass - sF.lag('Mass',1).over(w)).na.fill(0.0)

        self.set_param('with_field_mgain', True)

    def add_field_relation(self, progtable, hostmap):
        '''
        Add a field ('relation') in the gptable that defines the relation between 
        haloIdTarget and any halo that hosts a gas particle in the gptable.

        Parameters
        ----------
        haloIdTarget: int.
            The haloId of the halo in the current snapshot.
        gptable: pandas.DataFrame or Spark DataFrame
        progtable: pandas.DataFrame.
            Columns: haloId*, snapnum, progId, hostId, logMvir, logMsub
            Output of progen.find_all_previous_progenitors().
            Defines the progenitors of any halo in any previous snapshot.
        hostmap: pandas.DataFrame.
            Columns: snapnum*, haloId*, hostId
            Output of progen.build_haloId_hostId_map()
            Mapping between haloId and hostId for each snapshot.
        '''

        if(not self.validate_table(ValidateRuleWithField('relation', self.data.columns))):
            return

        # Find all halos that once hosted the particles
        halos = ProgTracker.compile_halo_hosts(self.data, ['snapnum','haloId'])

        # Update halos with the relation tag
        halos = ProgTracker.assign_relations_to_halos(halos, progtable, hostmap)
        
        gptable = pd.merge(gptable, halos, how='left',
                           left_on=['snapnum','haloId'], right_index=True)
        gptable['relation'] = gptable['relation'].fillna('IGM')

        self.set_param('with_field_relation', True)


    # TODO: This may better belong to some other module
    def find_particle_ancestors(pidlist, splittable):
        '''
        In case there is splitting, we need to find all the ancestors of a 
        gas particle in order to figure out where its wind material came from 
        before it spawned from its parent.
        Furthermore, we also need to find all the split events for any of the 
        parents.

        Parameters
        ----------
        pidlist: list.
            List of particle IDs for which to build the table
        splittable: pandas.DataFrame
            Table that contains information for all particle split events

        Returns
        -------
        ancestors: pandas.DataFrame
            Columns: PId, parentId, snapnext, gen
        '''

        # Find the maximum generation of a particle when it was created/spawned
        maxgen = splittable.groupby('parentId')['parentGen'].max().to_dict()
        
        df = splittable.loc[splittable.PId.isin(pidlist),
                            ['PId','parentId','Mass','snapnext','parentGen']]

        # At the time when the PId was spawned
        df['gen'] = df['PId'].map(maxgen).fillna(0).astype('int32') + 1
        df_next = df.copy()
        i, max_iter = 0, 10
        # Find the parent of parent until no more
        while(i < max_iter):
            i = i + 1
            # Does the parent have a grandparent? (not self).
            # [<PId>, parentId*] -> [PId*, parentId]
            df_next = pd.merge(df_next[['PId', 'parentId', 'gen', 'parentGen']],
                               splittable,
                               how='inner', left_on='parentId', right_on='PId',
                               suffixes=("_l", None))
            # If no more ancestors found, break the loop
            if(df_next.empty):
                break
            # How many generations have passed between the time when the parent
            # was spawned and the time it spawns the PId?
            df_next['gen'] = df_next.gen \
                + df_next['parentId_l'].map(maxgen).fillna(0).astype('int32') \
                - df_next['parentGen_l'] + 1
            df_next.drop(['parentId_l','parentGen_l','atime','PId'], axis=1, inplace=True)
            df_next.rename(columns={'PId_l':'PId'}, inplace=True)
            df = pd.concat([df, df_next], axis=0)
        if(i == max_iter):
            warnings.warn("Maximum iterations {} reached.".format(max_iter))

        # Self-splitting
        df_self = splittable.loc[splittable.parentId.isin(pidlist+list(df.parentId)),
                                 ['parentId','Mass','snapnext','parentGen']]
        # Does the parent have previous self-splitting?
        df_tmp = pd.merge(df[['PId','parentId','gen','parentGen']],
                          df_self[['parentId','parentGen','Mass','snapnext']],
                          how='inner', on='parentId',
                          suffixes=(None, "_r"))
        df_tmp = df_tmp.query('parentGen_r > parentGen')
        df_tmp['gen'] = df_tmp.gen \
            + df_tmp['parentGen_r'] - df_tmp['parentGen']
        df_tmp.drop(['parentGen_r'], axis=1, inplace=True)
            
        df_self['gen'] = df_self['parentGen']
        df_self['PId'] = df_self['parentId']

        # df_self contains particles out of the pidlist

        df = pd.concat([df, df_tmp, df_self], axis=0)
        # Sanity Check:
        # grp = df.groupby('PId')
        # grp.gen.count() - grp.gen.max() should be all 0
        return df[df.PId.isin(pidlist)]


# ----------------------------------------------------------------
#                      Class: PhEWPartTable
# ----------------------------------------------------------------

class PhewPartTable(TemporaryTable):
    '''
    Temporary table that contains selected PhEW particles and their histories.
    '''
    
    def __init__(self, model, snapnum, galIdTarget, **kwargs):
        super(PhEWPartTable, self).__init__(model, snapnum, galIdTarget)
        self._clsstr = "pptable"
        self._initialize()

        self._parse_keywords(kwargs)

        # Must match the following parameters to validate an existinig table
        self.validate_rule = ValidateRuleExisted(
            snapnum=self.snapnum,
            galId=self._galId,
            include_stars=self._pars.get('include_stars'),
            fformat=self._pars.get('fformat')
        )

        gptable = kwargs.get('gptable')
        if(gptable is None):
            self.load_gptable(**kwargs)
        else:
            self.gptable = gptable

    @classmethod
    def from_gptable(cls, gptable, **kwargs):
        '''
        Build the pptable directly from gptable.
        '''
        return cls(gptable.model, gptable.snapnum, gptable.galId,
                   gptable = gptable,
                   include_stars=gptable._pars.get('include_stars'))

    def _parse_keywords(self, **kwargs):
        '''
        Parse keywords specific to PhEWPartTable.
        '''

        if(kwargs.get('include_stars') is not None):
            self._pars['include_stars'] = kwargs['include_stars']
        if(kwargs.get('with_field_birthtag') is not None):
            self._pars['with_field_birthtag'] = kwargs['with_field_birthtag']
            

    def load_gptable(self):
        gptable = GasPartTable(self.model, self.snapnum, self.galId,
                               **kwargs)
        gptable.load_table()
        self.gptable = gptable

    def build_table(self, inittable, phewtable, spark=None, overwrite=False):
        '''
        Build or load a temporary table that stores all the necessary attributes 
        of selected PhEW particles that can be queried by the accretion tracking 
        engine.

        Parameters
        ----------
        inittable: pandas.DataFrame or Spark DataFrame
            Columns: PId*, snapfirst, minit, birthId, snaplast, mlast
            The initial/final status of all PhEW particles
            Needed to know from which halo a PhEW particles were born
            Created from Simulation.build_inittable_from_simulation()
        phewtable: pandas.DataFrame or Spark DataFrame
            Columns: PId*, snapnum, Mass, haloId, (Mloss)
            All PhEW particles in any snapshot of the simulation
            The function return is a subset of it
            Created from Simulation.build_phewtable_from_simulation()
        spark: SparkSession. Default=None
            if None, do not use spark.
        overwrite: boolean. Default=False
            If True, rebuild the table even if a file exists.

        Returns
        -------
        pptable: pandas.DataFrame or Spark DataFrame
            Columns: PId*, snapnum, haloId, Mloss, birthId
            Temporary table storing all the necessary attributes of selected PhEW
            particles that can be queried by the accretion tracking engine.
        '''

        if(overwrite==False and self.validate_table(self.validate_rule)):
            talk("Load existing {}.".format(self.clsstr), 'normal')
            self.load_table(spark=spark)
            return
        
        assert(self.gptable is not None)

        # Find all halos that ever hosted the gas particles in gptable
        halos = ProgTracker.compile_halo_hosts(self.gptable.data)

        # Find all PhEW particles that ever appeared in these halos
        pptable = pd.merge(halos, phewtable, how='inner',
                           left_on=['snapnum', 'haloId'],
                           right_on=['snapnum', 'haloId'])
        # pptable: snapnum, haloId, PId, Mloss

        # Add the birth halo information to pptable
        pptable = pd.merge(pptable, inittable[['PId','snapfirst','birthId']],
                           how='left', left_on='PId', right_on='PId')

        self.data = pptable

        self.set_params('with_field_birthtag', False)

    def add_field_birthtag(self, haloIdTarget, progtable, hostmap):
        '''
        Add a field ('birthTag') in the pptable that defines the relation between 
        haloIdTarget and the halo where a PhEW particle was born.

        Parameters
        ----------
        haloIdTarget: int.
            The haloId of the halo in the current snapshot.
        progtable: pandas.DataFrame.
            Columns: haloId*, snapnum, progId, hostId, logMvir, logMsub
            Output of progen.find_all_previous_progenitors().
            Defines the progenitors of any halo in any previous snapshot.
        hostmap: pandas.DataFrame.
            Columns: snapnum*, haloId*, hostId
            Output of progen.build_haloId_hostId_map()
            Mapping between haloId and hostId for each snapshot.

        '''

        if(not self.validate_table(ValidateRuleWithField('birthTag', self.data.columns))):
            return

        halos = ProgTracker.compile_halo_hosts(self.data, ['snapfirst','birthId'])
        halos = ProgTracker.assign_relations_to_halos(haloIdTarget, halos, progtable, hostmap)
        pptable = pd.merge(self.data, halos, how='left',
                           left_on=['snapfirst','birthId'], right_index=True)
        pptable.rename(columns={'relation':'birthTag'}, inplace=True)
        self.data = pptable

        self.set_param('with_field_birthtag', True)        


# ----------------------------------------------------------------
#                      Class: ProgTable
# ----------------------------------------------------------------

class ProgTable(PermanentTable):
    '''
    Permanent table that contains the halo assembly histories.
    '''
    
    def __init__(self, model, snapnum, **kwargs):
        super(ProgTable, self).__init__(model)
        self._clsstr = "progtable"
        self._snapnum = snapnum
        self._initialize()

        self._pars['snapnum'] = self._snapnum

        self._parse_keywords(**kwargs)

        # Must match the following parameters to validate an existinig table
        self.validate_rule = ValidateRuleExisted(
            snapnum=self._snapnum,
            fformat=self._pars.get('fformat')
        )

    def data(self):
        return self.data.set_index('haloId')

    def build_table(self, load_halo_mass=True, overwrite=False):
        '''
        Find the progenitors for all halos within a snapshot in all previous 
        snapshots.

        Parameters
        ----------
        load_halo_mass: boolean. Default=True
            If True. Load logMvir and logMsub for each progenitor.
        overwrite: boolean. Default=False.
            If False, first try to see if a table already exists. Create a new 
            table if not.
            If True, create a new table and overwrite the old one if needed.

        Returns
        -------
        progtable: pandas DataFrame.
            A pandas table storing information for the progenitors of halos at
            different time.
            columns: haloId*, snapnum, progId, hostId, logMvir, logMsub

        Examples
        --------
        >>> snap = snapshot.Snapshot('l12n144-phew', 100)
        >>> progtable = find_all_previous_progenitors(snap, overwrite=True)

        '''

        if(load_halo_mass == False):
            self._schema['columns'].remove('logMvir')
            self._schema['columns'].remove('logMsub')
            self._schema['columns'].append('Npart')        

        if(overwrite == False and self.validate_table(self.validate_rule)):
            talk("Load existing {} file: {}".format(self.clsstr, self.filename), 'normal')
            self.load_table()
            return

        talk("Finding progenitors for halos in snapshot_{:03d}".format(self.snapnum), 'normal')
        progtable = None
        snap = snapshot.Snapshot(self.model, self.snapnum)
        for snapnum in tqdm(range(0, self.snapnum+1), desc='snapnum', ascii=True):
            snapcur = snapshot.Snapshot(self.model, snapnum, verbose='talky')
            if(load_halo_mass):
                snapcur.load_halos(['Mvir', 'Msub'])
            else:
                snapcur.load_halos(['Npart'])            
            haloId2hostId = galaxy.read_sopar(snapcur._path_sopar, as_dict=True)
            haloId2progId = self.find_progenitors(snap, snapcur)
            haloId2hostId[0] = 0
            df = pd.DataFrame(index=haloId2progId.keys())
            df['snapnum'] = snapnum
            df['progId'] = df.index.map(haloId2progId)
            df['hostId'] = df.progId.map(haloId2hostId)
            df = pd.merge(df, snapcur.halos, how='left', left_on='progId', right_index=True)
            progtable = df.copy() if (progtable is None) else pd.concat([progtable, df])
        progtable.index.rename('haloId', inplace=True)
        if(load_halo_mass == False):
            progtable.Npart = progtable.Npart.fillna(0).astype('int')
        self.data = progtable

    def find_progenitors(self, snap, snap_early):
        '''
        For each halo from a snapshot, find its main progenitor in some early 
        snapshot. The main progenitor of a halo is defined as the halo that hosts 
        a majority of its current dark matter particles.

        Algorithm:
        1. Select dark particles that was found in a halo in both the early and the current snapshot (hostId > 0)
        2. Find the total number of dark particles, in any hostId that was from progId
        3. Pick the progId with max(count) as the progenitor of hostId

        Parameters
        ----------
        snap: class Snapshot.
        snap_early: class Snapshot.
            A snapshot at some earlier time in which the progenitors are found.

        Returns
        -------
        haloId2progId: dict.
            A temporary mapping from snap.haloId to snap_early.progId.
            Note: not all haloId will find a progenitor.
        '''

        assert(snap.snapnum >= snap_early.snapnum), "snap_early must be at an earlier time than snap."
        if(snap.snapnum == snap_early.snapnum):
            haloId2progId = {i:i for i in range(1, snap.ngals+1)}
            return haloId2progId

        snap.load_dark_particles()
        snap_early.load_dark_particles()
        dp = snap.dp[snap.dp.haloId > 0].set_index('PId')
        dpe = snap_early.dp[snap_early.dp.haloId > 0]\
                        .set_index('PId')\
                        .rename(columns={'haloId':'progId'})
        # dpe could be empty if no galaxy has formed yet.

        dp = pd.merge(dp, dpe, how='inner', left_index=True, right_index=True)
        grp = dp.reset_index().groupby(['haloId','progId']).count().reset_index()
        idx = grp.groupby('haloId')['PId'].transform(max) == grp['PId']
        haloId2progId = dict(zip(grp[idx].haloId, grp[idx].progId))

        # If no progenitor is found, set progId to 0
        for i in range(1, snap.ngals+1):
            if(haloId2progId.get(i) is None):
                haloId2progId[i] = 0
        return haloId2progId

    @property
    def snapnum(self):
        return self._snapnum

    @property
    def filename(self):
        return self._filename

    @filename.setter
    def filename(self, name):
        if(name == ""):
            name = "{}_{:03d}.{}".format(self.clsstr, self._snapnum, self._fformat)
        self._filename = name

# ----------------------------------------------------------------
#                      Class: SplitTable
# ----------------------------------------------------------------

class SplitTable(PermanentTable):
    '''
    Permanent table that contains the splitting events.
    '''
    
    def __init__(self, model, **kwargs):
        super(SplitTable, self).__init__(model)
        self._path_out = self._simulation._path_data
        
        self._clsstr = "splittable"
        self._initialize()

        self._parse_keywords(**kwargs)

        # Must match the following parameters to validate an existinig table
        self.validate_rule = ValidateRuleExisted(
            fformat=self._pars.get('fformat')
        )

    def data(self):
        return self.data.set_index('haloId')

    def build_table(self, overwrite=False):
        '''

        Parameters
        ----------
        overwrite: boolean. Default=False.
            If True, force creating the table.
        spark: SparkSession. Default=None
            If None, return the table as a pandas dataframe.
            Otherwise, return the table as a Spark dataframe.
        
        Returns
        -------
        splittable: pandas.DataFrame or Spark DataFrame
            Columns: PId*, parentId, Mass, atime, snapnext
            snapnext is the snapshot AFTER the split happened
        '''
        # Load if existed.
        if(overwrite == False and self.validate_table(self.validate_rule)):
            talk("Loading existing {} file: {}".format(self.clsstr, self.filename), 'normal')
            self.load_table()
            return

        # Create new if not existed.
        df = winds.read_split(self._path_winds)
        redz = Simulation.load_timeinfo_for_snapshots()
        df['snapnext'] = df.atime.map(lambda x : bisect_right(redz.a, x))

        # Now do something fansy: get the generation (reverse order) of
        # each splitting event.
        parents = df.groupby('parentId')
        df['parentGen'] = df.groupby('parentId')['atime']\
                      .rank(ascending=False)\
                      .astype('int32')

        df = df[schema_splittable['columns']]
        self.data = df

        
# ----------------------------------------------------------------
#                      Class: HostTable
# ----------------------------------------------------------------

class HostTable(PermanentTable):
    '''
    Permanent table that defines the host halos.
    '''
    
    def __init__(self, model, **kwargs):
        super(HostTable, self).__init__(model)
        self._clsstr = "hosttable"
        self._initialize()

        self._parse_keywords(**kwargs)

        # Must match the following parameters to validate an existinig table
        self.validate_rule = ValidateRuleExisted(
            fformat=self._pars.get('fformat')
        )

    def build_table(self, overwrite=False):
        '''
        Build maps between haloId and hostId for each snapshot before the 
        snapshot parsed in the arguments. 
        Calls progen.build_haloId_hostId_ map(snap, overwrite)

        Returns
        -------
        hosttable: pandas.DataFrame
            Columns: snapnum*, haloId*, hostId
        '''
        from progen import ProgTracker
        
        hosttable = ProgTracker.build_haloId_hostId_map(self._simulation)
        self.data = hosttable


# ----------------------------------------------------------------
#                      Class: InitTable
# ----------------------------------------------------------------

class InitTable(PermanentTable):
    '''
    Permanent table that contains the splitting events.
    '''
    
    def __init__(self, model, **kwargs):
        super(InitTable, self).__init__(model)
        self._path_out = self._simulation._path_data
        
        self._clsstr = "inittable"
        self._initialize()        

        self._parse_keywords(**kwargs)

        # Must match the following parameters to validate an existinig table
        self.validate_rule = ValidateRuleExisted(
            fformat=self._pars.get('fformat')
        )

    def build_table(self, overwrite=False, spark=None):
        '''
        Gather all PhEW particles within the entire simulation and find their 
        initial and final attributes. Returns an inittable that will be queried
        extensively by the wind tracking engine.

        Parameters
        ----------
        overwrite: boolean. Default=False.
            If True, force creating the table.
        spark: SparkSession. Default=None
            If None, return the table as a pandas dataframe.
            Otherwise, return the table as a Spark dataframe.
        
        Returns
        -------
        inittable: pandas.DataFrame or Spark DataFrame
            The initial and final attributes of all PhEW particles
            Columns: PId*, snapfirst, minit, birthId, snaplast, mlast
            snapfirst is the snapshot BEFORE it was launched as a wind
            snaplast is the snapshot AFTER it stopped being a PhEW
        '''

        # Load if existed.
        if(overwrite == False and self.validate_table(self.validate_rule)):
            talk("Loading existing {} file: {}".format(self.clsstr, self.filename), 'normal')
            self.load_table()
            return
        
        # Create new if not existed.
        dfi = winds.read_initwinds(self._path_winds, columns=['atime','PhEWKey','Mass','PID'], minPotIdField=True)
        redz = Simulation.load_timeinfo_for_snapshots()
        dfi['snapfirst'] = dfi.atime.map(lambda x : bisect_right(redz.a, x)-1)
        dfi.rename(columns={'Mass':'minit','PID':'PId'}, inplace=True)
        grps = dfi.groupby('snapfirst')
        
        talk("Looking for the halos where PhEW particles are born.", "normal")
        frames = []
        for snapnum in tqdm(range(0, sim.nsnaps), desc='snapnum', ascii=True):
            snap = snapshot.Snapshot(sim.model, snapnum)
            snap.load_gas_particles(['PId','haloId'])
            try:
                grp = grps.get_group(snapnum)
            except:
                continue
            df = pd.merge(grp, snap.gp, how='left',
                          left_on='PId', right_on='PId')
            frames.append(df)
        # TODO: Use MinPotId to find more accurate birthId
        dfi = pd.concat(frames, axis=0).rename(columns={'haloId':'birthId'})
        dfi.birthId = dfi.birthId.fillna(0).astype('int32')

        # Load final status of the winds and merge
        dfr = winds.read_rejoin(self._path_winds, columns=['atime','PhEWKey','Mass'])
        dfr['snaplast'] = dfr.atime.map(lambda x : bisect_right(redz.a, x))
        dfr.rename(columns={'Mass':'mlast'}, inplace=True)
        df = pd.merge(dfi, dfr, how='left', left_on='PhEWKey', right_on='PhEWKey')
        df.snaplast = df.snaplast.fillna(109).astype('int32')
        df.mlast = df.mlast.fillna(0.0).astype('float32')

        df = df[['PId','snapfirst','minit','birthId','snaplast','mlast']]

        self.data = df

# ----------------------------------------------------------------
#                      Class: PhEWTable
# ----------------------------------------------------------------

class PhEWTable(PermanentTable):
    '''
    Permanent table that contains all PhEW particles.
    '''
    
    def __init__(self, model, **kwargs):
        super(PhEWTable, self).__init__(model)
        self._path_out = self._simulation._path_data
        self._fformat = "parquet"

        self._clsstr = "phewtable"
        self._initialize()        

        self._parse_keywords(**kwargs)

        # Must match the following parameters to validate an existinig table
        self.validate_rule = ValidateRuleExisted(
            ignore_init=self._pars.get('ignore_init'),
            fformat=self._pars.get('fformat')
        )

    def _parse_keywords(self, **kwargs):
        '''
        Parse keywords specific to PermanentTable
        '''
        if(kwargs.get('fformat') is not None):
            self._pars['fformat'] = kwargs['fformat']
        else:
            self._pars['fformat'] = 'parquet'

    def load_table(self, spark=None):
        '''
        Load a gigantic table that contains all PhEW particles ever appeared in a
        simulation. The initial status and the final status of a PhEW particle is
        found in initwinds.* and rejoin.* files. Any one record corresponds to a 
        PhEW particle at a certain snapshot.

        Parameters
        ----------
        spark: SparkSession. Default=None
            If None, return the table as a pandas dataframe.
            Otherwise, return the table as a Spark dataframe.

        Returns
        -------
        phewtable: pandas.DataFrame or Spark DataFrame
            A gigantic table containing all PhEW particles in any snapshot.
            Columns: PId*, snapnum, Mass, haloId
            This table will be heavily queried by the accretion tracking engine.
        '''

        assert(os.path.exists(self._path_table)), "phewtable.parquet file is not found. Use Simulation.build_phewtable() to create one."

        schemaParquet = utils.read_parquet_schema(self._path_table)
        if('Mloss' in set(schemaParquet.column)):
            talk("Loading complete phewtable.parquet file.", "normal")
        else:
            talk("The phewtable.parquet file still misses the 'Mloss' field. Use Simulation.compute_mloss_partition_by_pId() to patch the file.", "normal")
        
        if(spark is not None):
            self.data = spark.read.parquet(self._path_table)
        else:
            phewtable = pd.read_parquet(self._path_table)
            self.data = phewtable

    def build_table(self, snaplast=None, overwrite=False, ignore_init=False, spark=None):
        '''
        Build a gigantic table that contains all PhEW particles ever appeared in a
        simulation. The initial status and the final status of a PhEW particle is
        found in initwinds.* and rejoin.* files. Any one record corresponds to a 
        PhEW particle at a certain snapshot.
        Opt for parallel processing.

        Parameters
        ----------
        snaplast: int. Default=None
            If None. Use all snapshot until the end of the simulation.
        overwrite: boolean. Default=False
            If True, force creating the table and overwrite existing file.
        ignore_init: boolean. Default=False
            If True, do not include the initial/final state of the particles
            in the phewtable.
        spark: SparkSession. Default=None
            If None, return the table as a pandas dataframe.
            Otherwise, return the table as a Spark dataframe.

        Returns
        -------
        phewtable: pandas.DataFrame or Spark DataFrame
            A gigantic table containing all PhEW particles in any snapshot.
            Columns: PId*, snapnum, Mass, haloId
            This table will be heavily queried by the accretion tracking engine.

        OutputFiles
        -----------
        data/phewtable.parquet
        '''

        # Load if existed.
        if(overwrite == False and self.validate_table(self.validate_rule)):
            talk("Loading existing {} file: {}".format(self.clsstr, self.filename), 'normal')
            self.load_table()
            return

        talk("Building {} for model: {}".format(self.filename, self.model), "talky")
        
        if(snaplast is None):
            snaplast = self.nsnaps
            
        phewtable = None
        for snapnum in tqdm(range(0, snaplast), desc='snapnum', ascii=True):
            snap = snapshot.Snapshot(sim.model, snapnum, verbose='talky')
            snap.load_gas_particles(['PId','Mass','Mc','haloId'])
            gp = snap.gp.loc[snap.gp.Mc > 0, ['PId','Mass','haloId']]
            if(not gp.empty):
                gp.loc[:,'snapnum'] = snapnum
            phewtable = gp.copy() if phewtable is None else pd.concat([phewtable, gp])

        if (not ignore_init):
            inittable = self.build_inittable()
            tmp = inittable[['PId','minit','snapfirst']]\
                .rename(columns={'minit':'Mass','snapfirst':'snapnum'})
            tmp['haloId'] = 0 # Unknown
            phewtable = pd.concat([phewtable, tmp])
            tmp = inittable[['PId','mlast','snaplast']]\
                .rename(columns={'mlast':'Mass','snaplast':'snapnum'})
            tmp['haloId'] = 0 # Unknown        
            phewtable = pd.concat([phewtable, tmp])

        # Very computationally intensive
        phewtable = phewtable.sort_values(['PId','snapnum'])
        phewtable.snapnum = phewtable.snapnum.astype('int')

        # Write parquet file.
        self.data = phewtable

        # New table is clean without the derived fields
        self.set_param("ignore_init", ignore_init)
        self.set_param('with_field_mloss', False)

        # schema['columns'].remove('Mloss')

    def add_field_mloss(self, spark=None):
        '''
        For each PhEW particle in the phewtable, compute its mass loss since last
        snapshot (it could have just launched after the last snapshot). It will 
        replace the phewtable.parquet file.

        Parameters
        ----------
        overwrite: boolean. Default=False
            If True, force creating the table and overwrite existing file.
        spark: SparkSession. Default=None
            If None, use pandas.
            Otherwise, use Spark.

        Returns
        -------
        None

        OutputFiles
        -----------
        data/phewtable.parquet
        '''

        if(self._fformat == "parquet"):
            schemaParquet = utils.read_parquet_schema(self._path_table)
            columns = set(schemaParquet.column)
        elif(self._fformat == "csv"):
            columns = self.data.columns
        
        if(self.validate_table(ValidateRuleWithField('Mloss', columns))):
            talk("The {} file is already complete. To recompute the field, set overwrite=True.".format(self.filename), "talky")
            return

        talk("Computing the Mloss field for PhEW particles.", "normal")

        if(spark is None):
            phewtable = self.load_table()
            # This is a very expensive operation
            phewtable['Mloss'] = phewtable.groupby('PId').diff().fillna(0.0)
            self._schema['columns'].append('Mloss')
            self.data = phewtable
            self.save_table()
        else:
            phewtable = self.load_phewtable(spark=spark)            
            w = Window.partitionBy(phewtable.PId).orderBy(phewtable.snapnum)
            phewtable = phewtable.withColumn('Mloss', phewtable.Mass - sF.lag('Mass',1).over(w)).na.fill(0.0)
            self.data = phewtable
            self.save_table(spark=spark)

if(__name__ == "__main__"):
    hosttable = HostTable('l25n144-test')
    inittable = InitTable('l25n144-test')
    progtable = ProgTable('l25n144-test', snapnum=108)
    phewtable = PhEWTable('l25n144-test')
    splittable = SplitTable('l25n144-test')    
