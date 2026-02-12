"""
Sorting logic.

Sorts variables and coordinates alphabetically for consistent output presentation.
"""

from ...base import StageContext


class Sorting:
    """
    Sort variables and coordinates in the dataset alphabetically.
    
    This is a cosmetic step that ensures consistent variable ordering
    in the output. Variables with the same base name (e.g., temperature,
    temperature_1) are naturally grouped together by alphabetical sorting.
    
    This step should typically run last since it
    only affects presentation, not data content.
    """
    
    def process(self, context: StageContext) -> StageContext:
        """
        Sort all variables and coordinates alphabetically.
        
        Parameters
        ----------
        context : StageContext
            The processing context.
        
        Returns
        -------
        StageContext
            Updated context with sorted dataset.
        """
        ds = context.dataset
        
        # Get all variable and coordinate names, sorted
        all_names = sorted(list(ds.data_vars) + list(ds.coords))
        
        # Create new dataset with sorted order
        ds_sorted = ds[all_names]
        
        # Preserve attributes
        ds_sorted.attrs = ds.attrs.copy()
        
        context.dataset = ds_sorted
        
        return context
