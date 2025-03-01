"""Component registry for MemoryWeave.

This module provides the registry for pipeline components,
allowing components to be registered, retrieved, and managed.
"""

import logging
from typing import Dict, List, Optional, Set

from memoryweave.interfaces.pipeline import (
    ComponentID,
    ComponentType,
    IComponent,
    IComponentRegistry,
)


class ComponentRegistry(IComponentRegistry):
    """Registry for pipeline components."""

    def __init__(self):
        """Initialize the component registry."""
        self._components: Dict[ComponentID, IComponent] = {}
        self._type_index: Dict[ComponentType, Set[ComponentID]] = {
            component_type: set() for component_type in ComponentType
        }
        self._logger = logging.getLogger(__name__)

    def register(self, component: IComponent) -> None:
        """Register a component in the registry."""
        component_id = component.get_id()
        component_type = component.get_type()

        # Check if component with same ID already exists
        if component_id in self._components:
            self._logger.warning(
                f"Component with ID '{component_id}' already registered. Overwriting."
            )

        # Register component
        self._components[component_id] = component
        self._type_index[component_type].add(component_id)

        self._logger.debug(
            f"Registered component '{component_id}' of type '{component_type.name}'"
        )

    def get_component(self, component_id: ComponentID) -> Optional[IComponent]:
        """Get a component by ID."""
        return self._components.get(component_id)

    def get_components_by_type(self, component_type: ComponentType) -> List[IComponent]:
        """Get all components of a specific type."""
        component_ids = self._type_index.get(component_type, set())
        return [self._components[component_id] for component_id in component_ids]

    def clear(self) -> None:
        """Clear all components from the registry."""
        self._components.clear()
        for component_type in ComponentType:
            self._type_index[component_type].clear()

        self._logger.debug("Cleared component registry")

    def remove(self, component_id: ComponentID) -> bool:
        """Remove a component from the registry.
        
        Returns:
            True if the component was removed, False if it wasn't found
        """
        if component_id not in self._components:
            return False

        # Get component type
        component_type = self._components[component_id].get_type()

        # Remove from indexes
        del self._components[component_id]
        self._type_index[component_type].remove(component_id)

        self._logger.debug(f"Removed component '{component_id}'")
        return True

    def get_all_components(self) -> List[IComponent]:
        """Get all registered components."""
        return list(self._components.values())

    def get_component_count(self) -> int:
        """Get the number of registered components."""
        return len(self._components)

    def has_component(self, component_id: ComponentID) -> bool:
        """Check if a component with the given ID is registered."""
        return component_id in self._components

    def get_component_ids(self) -> List[ComponentID]:
        """Get all registered component IDs."""
        return list(self._components.keys())

    def get_component_ids_by_type(self, component_type: ComponentType) -> List[ComponentID]:
        """Get all component IDs of a specific type."""
        return list(self._type_index.get(component_type, set()))
