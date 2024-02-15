from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.models import Base, Mixtures, MixtureProperties, PhaseEquilibrium, Components

engine = create_engine('sqlite:///chemicals.db')
Session = sessionmaker(bind=engine)

def create_database():
    Base.metadata.create_all(engine)

def add_mixture(name, formula):
    session = Session()
    new_mixture = Mixtures(name=name, formula=formula)
    session.add(new_mixture)
    session.commit()
    session.close()

def add_phase_equilibrium(mixture_id, dataframe):
    session = Session()
    equilibrium_objects = [
        PhaseEquilibrium(
            mixture_id=mixture_id,
            x=row['x'],
            y=row['y'],
            t=row['t'],
            p=row['P']
        )
        for index, row in dataframe.iterrows()
    ]
    session.bulk_save_objects(equilibrium_objects)
    session.commit()
    session.close()

def add_component(name, formula):
    session = Session()
    new_component = Components(name=name, formula=formula)
    session.add(new_component)
    session.commit()
    session.close()

def search_mixtures(search_query):
    session = Session()
    results = session.query(Mixtures).filter(Mixtures.name.like(f'%{search_query}%')).all()
    session.close()
    return results
