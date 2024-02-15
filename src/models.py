from sqlalchemy import Column, Integer, String, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class Mixtures(Base):
    __tablename__ = 'mixtures'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    formula = Column(String)
    properties = relationship("MixtureProperties", back_populates="mixture", cascade="all, delete")

class MixtureProperties(Base):
    __tablename__ = 'mixture_properties'
    id = Column(Integer, primary_key=True)
    type = Column(String)  # Тип свойства, например, "Фазовое равновесие"
    mixture_id = Column(Integer, ForeignKey('mixtures.id'))
    mixture = relationship("Mixtures", back_populates="properties")
    property_values = relationship("PropertyValue", back_populates="mixture_property")

    __mapper_args__ = {
        'polymorphic_identity': 'mixture_properties',
        'polymorphic_on': type
    }

class PhaseEquilibrium(MixtureProperties):
    __tablename__ = 'phase_equilibrium'
    id = Column(Integer, ForeignKey('mixture_properties.id'), primary_key=True)
    x = Column(Float)
    y = Column(Float)
    t = Column(Float)
    p = Column(Float)

    __mapper_args__ = {
        'polymorphic_identity': 'phase_equilibrium',
    }

class Components(Base):
    __tablename__ = 'components'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    formula = Column(String)
    component_properties = relationship("ComponentProperties", back_populates="component", cascade="all, delete")

class ComponentProperties(Base):
    __tablename__ = 'component_properties'
    id = Column(Integer, primary_key=True)
    component_id = Column(Integer, ForeignKey('components.id'))
    component = relationship("Components", back_populates="component_properties")
    property_values = relationship("PropertyValue", back_populates="component_property")

class PropertyValue(Base):
    __tablename__ = 'property_values'
    id = Column(Integer, primary_key=True)
    component_property_id = Column(Integer, ForeignKey('component_properties.id'))
    mixture_property_id = Column(Integer, ForeignKey('mixture_properties.id'))
    x = Column(Float)
    y = Column(Float)
    t = Column(Float)
    p = Column(Float)
    mixture_property = relationship("MixtureProperties", back_populates="property_values")
    component_property = relationship("ComponentProperties", back_populates="property_values")
