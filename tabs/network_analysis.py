import networkx as nx
import matplotlib.pyplot as plt
import random
import pandas as pd
import seaborn as sns

import plotly.graph_objects as go
import pandas as pd
import numpy as np
import matplotlib.colors as mcolors
import base64
import io
from PIL import Image, ImageDraw

def create_gender_org_chart(org_data=None, width=800, height=600):
    """
    Create an organizational chart visualizing gender distribution with a color gradient.
    
    Args:
        org_data (dict, optional): Dictionary containing organizational data with keys 'id', 
                                'parent', 'name', and 'gender_ratio'. If None, sample data is used.
        width (int): Width of the figure in pixels
        height (int): Height of the figure in pixels
        
    Returns:
        plotly.graph_objects.Figure: The interactive organizational chart
    """
    # Use sample data if none provided
    if org_data is None:
        org_data = {
            "id": ["Exec", "Eng", "Mktg", "HR", "Prod", 
                "FE", "BE", "DevOps", 
                "Digital", "Brand", 
                "Recruit", "Train",
                "Design", "PM", "Research"],
            "parent": ["", "Exec", "Exec", "Exec", "Exec", 
                    "Eng", "Eng", "Eng", 
                    "Mktg", "Mktg", 
                    "HR", "HR",
                    "Prod", "Prod", "Prod"],
            "name": ["Executive", "Engineering", "Marketing", "HR", "Product", 
                    "Frontend", "Backend", "DevOps", 
                    "Digital", "Brand", 
                    "Recruiting", "Training",
                    "Design", "Product Management", "Research"],
            "gender_ratio": [0.35, 0.28, 0.63, 0.81, 0.45, 
                            0.32, 0.21, 0.18, 
                            0.58, 0.72, 
                            0.75, 0.85,
                            0.62, 0.48, 0.55]
        }

    # Convert to DataFrame
    df = pd.DataFrame(org_data)

    # Create a gradient color function from blue (male) to pink (female)
    def get_color(ratio):
        # Blue (male) to pink (female) gradient
        blue = "#1E88E5"  # Male
        neutral = "#A9A9A9"  # Neutral gray for balanced
        pink = "#D81B60"  # Female
        
        # A more reliable approach using manual color interpolation
        if ratio == 0.5:
            return neutral
        elif ratio < 0.5:
            # Scale ratio to 0-1 range for blue to neutral
            scaled_ratio = ratio / 0.5
            # Convert hex to RGB
            blue_rgb = mcolors.colorConverter.to_rgb(blue)
            neutral_rgb = mcolors.colorConverter.to_rgb(neutral)
            # Linear interpolation between colors
            r = blue_rgb[0] * (1 - scaled_ratio) + neutral_rgb[0] * scaled_ratio
            g = blue_rgb[1] * (1 - scaled_ratio) + neutral_rgb[1] * scaled_ratio
            b = blue_rgb[2] * (1 - scaled_ratio) + neutral_rgb[2] * scaled_ratio
            # Convert back to hex
            return mcolors.to_hex((r, g, b))
        else:
            # Scale ratio to 0-1 range for neutral to pink
            scaled_ratio = (ratio - 0.5) / 0.5
            # Convert hex to RGB
            neutral_rgb = mcolors.colorConverter.to_rgb(neutral)
            pink_rgb = mcolors.colorConverter.to_rgb(pink)
            # Linear interpolation between colors
            r = neutral_rgb[0] * (1 - scaled_ratio) + pink_rgb[0] * scaled_ratio
            g = neutral_rgb[1] * (1 - scaled_ratio) + pink_rgb[1] * scaled_ratio
            b = neutral_rgb[2] * (1 - scaled_ratio) + pink_rgb[2] * scaled_ratio
            # Convert back to hex
            return mcolors.to_hex((r, g, b))

    # Add color column to the DataFrame
    df['color'] = df['gender_ratio'].apply(get_color)
    df['hover_text'] = df.apply(lambda row: f"{row['name']}<br>{int(row['gender_ratio']*100)}% Female / {int((1-row['gender_ratio'])*100)}% Male", axis=1)

    # Create a custom hierarchical layout
    def create_hierarchical_layout(df):
        # Create a dictionary to store node positions
        pos = {}
        
        # First, identify all levels
        levels = {}
        level = 0
        
        # Start with nodes that have no parent
        current_level_nodes = df[df['parent'] == '']['id'].tolist()
        
        while current_level_nodes:
            levels[level] = current_level_nodes
            next_level_nodes = []
            
            for node in current_level_nodes:
                # Find all children of this node
                children = df[df['parent'] == node]['id'].tolist()
                next_level_nodes.extend(children)
            
            current_level_nodes = next_level_nodes
            level += 1
        
        # Calculate positions for each level
        for level_num, nodes in levels.items():
            num_nodes = len(nodes)
            
            for i, node in enumerate(nodes):
                # Distribute nodes horizontally
                x_pos = (i - (num_nodes - 1) / 2) * 100
                # Place levels vertically
                y_pos = -level_num * 150
                
                pos[node] = (x_pos, y_pos)
        
        return pos, levels

    # Helper function to create gradient legend bar
    def create_gradient_bar(height=200, width=20):
        # Create a gradient image
        image = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(image)
        
        # Draw gradient from pink to blue
        for y in range(height):
            # Calculate ratio based on position
            ratio = 1 - (y / height)
            color = get_color(ratio)
            # Remove # from hex color
            rgb_color = tuple(int(color[1:][i:i+2], 16) for i in (0, 2, 4))
            draw.line([(0, y), (width, y)], fill=rgb_color)
        
        # Convert to base64 string for embedding
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        img_str = 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()
        
        return img_str

    # Get positions of nodes
    pos, levels = create_hierarchical_layout(df)

    # Create edge traces
    edge_x = []
    edge_y = []

    for _, row in df.iterrows():
        if row['parent'] != '':  # If the node has a parent
            child_id = row['id']
            parent_id = row['parent']
            
            x0, y0 = pos[parent_id]
            x1, y1 = pos[child_id]
            
            # Draw curved lines for better visualization
            control_x = x0
            control_y = (y0 + y1) / 2
            
            # Create points for a quadratic bezier curve
            curve_points = 20
            for i in range(curve_points + 1):
                t = i / curve_points
                # Quadratic bezier formula
                x = (1-t)**2 * x0 + 2*(1-t)*t * control_x + t**2 * x1
                y = (1-t)**2 * y0 + 2*(1-t)*t * control_y + t**2 * y1
                
                edge_x.append(x)
                edge_y.append(y)
            
            # Add None to separate line segments
            edge_x.append(None)
            edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    # Create node traces
    node_x = []
    node_y = []
    node_colors = []
    node_text = []
    node_sizes = []

    for index, row in df.iterrows():
        node_id = row['id']
        if node_id in pos:
            x, y = pos[node_id]
            node_x.append(x)
            node_y.append(y)
            node_colors.append(row['color'])
            node_text.append(row['hover_text'])
            
            # Make nodes at lower levels (higher in org) larger
            level = [lvl for lvl, nodes in levels.items() if node_id in nodes][0]
            node_sizes.append(30 - level * 3)  # Decrease size for deeper levels

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=False,
            color=node_colors,
            size=node_sizes,
            line_width=2))

    # Add node labels
    node_labels = []
    for index, row in df.iterrows():
        node_id = row['id']
        if node_id in pos:
            x, y = pos[node_id]
            node_labels.append(
                dict(
                    x=x,
                    y=y,
                    text=row['name'],
                    showarrow=False,
                    font=dict(size=10),
                    xanchor="center",
                    yanchor="bottom",
                    yshift=15
                )
            )

    # Create the figure for org chart
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Organizational Gender Distribution',
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=node_labels,
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    ))

    # Add a color scale legend in the top right
    fig.add_layout_image(
        dict(
            source=create_gradient_bar(height=200, width=20),
            xref="paper", yref="paper",
            x=1.02, y=1,
            sizex=0.05, sizey=0.4,
            xanchor="left", yanchor="top"
        )
    )

    # Add annotation for the legend
    fig.add_annotation(
        x=1.045, y=1,
        xref="paper", yref="paper",
        text="100% Female",
        showarrow=False,
        xanchor="left", yanchor="top",
        font=dict(size=10)
    )

    fig.add_annotation(
        x=1.045, y=0.6,
        xref="paper", yref="paper",
        text="100% Male",
        showarrow=False,
        xanchor="left", yanchor="bottom",
        font=dict(size=10)
    )

    fig.add_annotation(
        x=1.045, y=0.8,
        xref="paper", yref="paper",
        text="50-50",
        showarrow=False,
        xanchor="left", yanchor="middle",
        font=dict(size=10)
    )

    # Make figure wider to accommodate the legend
    fig.update_layout(
        width=width,
        height=height,
        margin=dict(r=100)
    )
    
    return fig

def create_org_chart():
    """
    Creates and visualizes an organizational chart with employee hierarchy.
    
    Returns:
        G: NetworkX DiGraph object representing the organizational structure
    """
    # Create an empty directed graph
    G = nx.DiGraph()

    # Define employee data with attributes
    employees = [
        # Executives
        {"id": 1, "name": "Sarah Johnson", "gender": "female", "experience": "exec", 
        "designation": "CEO", "reports_to": None},
        
        # Senior Management
        {"id": 2, "name": "David Chen", "gender": "male", "experience": "senior", 
        "designation": "CTO", "reports_to": 1},
        {"id": 3, "name": "Maria Rodriguez", "gender": "female", "experience": "senior", 
        "designation": "CFO", "reports_to": 1},
        {"id": 4, "name": "James Wilson", "gender": "male", "experience": "senior", 
        "designation": "COO", "reports_to": 1},
        
        # Mid-level Management
        {"id": 5, "name": "Priya Patel", "gender": "female", "experience": "mid", 
        "designation": "Engineering Director", "reports_to": 2},
        {"id": 6, "name": "Michael Kim", "gender": "male", "experience": "mid", 
        "designation": "Finance Director", "reports_to": 3},
        {"id": 7, "name": "Emily Brown", "gender": "female", "experience": "mid", 
        "designation": "Operations Manager", "reports_to": 4},
        {"id": 8, "name": "Carlos Mendez", "gender": "male", "experience": "mid", 
        "designation": "Product Director", "reports_to": 2},
        
        # Entry/Mid Level
        {"id": 9, "name": "Lisa Wong", "gender": "female", "experience": "mid", 
        "designation": "Senior Developer", "reports_to": 5},
        {"id": 10, "name": "John Smith", "gender": "male", "experience": "mid", 
        "designation": "Senior Accountant", "reports_to": 6},
        {"id": 11, "name": "Raj Gupta", "gender": "male", "experience": "entry", 
        "designation": "Developer", "reports_to": 9},
        {"id": 12, "name": "Emma Davis", "gender": "female", "experience": "entry", 
        "designation": "Developer", "reports_to": 9},
        {"id": 13, "name": "Samuel Park", "gender": "male", "experience": "entry", 
        "designation": "Junior Accountant", "reports_to": 10},
        {"id": 14, "name": "Alex Johnson", "gender": "male", "experience": "entry", 
        "designation": "Operations Associate", "reports_to": 7},
        {"id": 15, "name": "Olivia Martinez", "gender": "female", "experience": "entry", 
        "designation": "Product Associate", "reports_to": 8}
    ]

    # Add nodes with attributes
    for employee in employees:
        G.add_node(employee["id"], 
                name=employee["name"],
                gender=employee["gender"],
                experience=employee["experience"],
                designation=employee["designation"])

    # Add edges for reporting relationships
    for employee in employees:
        if employee["reports_to"] is not None:
            G.add_edge(employee["reports_to"], employee["id"])

    # Calculate reportees for each employee
    for node in G.nodes():
        reportees = list(G.successors(node))
        G.nodes[node]["reportees"] = reportees
        G.nodes[node]["num_reportees"] = len(reportees)

    # Hierarchical layout function
    def hierarchy_pos(G, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
        """
        If the graph is a tree this will return the positions to plot this in a 
        hierarchical layout.
        """
        if root is None:
            if isinstance(G, nx.DiGraph):
                root = next(iter(nx.topological_sort(G)))  # root is the first node in topological sort
            else:
                root = random.choice(list(G.nodes))

        def _hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None):
            """Recursive function for positioning nodes in hierarchical layout."""
            if pos is None:
                pos = {root: (xcenter, vert_loc)}
            else:
                pos[root] = (xcenter, vert_loc)
            
            children = list(G.neighbors(root))
            if parent is not None:
                children.remove(parent)  # This removes the edge going back to the parent (if there's one)
            if not children:  # If no children, we're done
                return pos
                
            # Count immediate children and get positions
            dx = width / len(children) 
            nextx = xcenter - width/2 - dx/2
            
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G, child, width=dx, vert_gap=vert_gap, 
                                    vert_loc=vert_loc-vert_gap, xcenter=nextx,
                                    pos=pos, parent=root)
            return pos
        
        return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)

    # Visualization settings
    plt.figure(figsize=(14, 10))

    # Find the root node (CEO) - the one with no incoming edges
    root = [node for node, in_degree in G.in_degree() if in_degree == 0][0]

    # Create an undirected copy of the graph for layout calculation
    layout_graph = G.to_undirected()

    # Use our fixed hierarchy layout function
    pos = hierarchy_pos(layout_graph, root=root)

    # Define colors based on gender
    node_colors = ['skyblue' if G.nodes[n]['gender'] == 'male' else 'lightpink' for n in G.nodes()]

    # Define node sizes based on experience level
    exp_size = {'entry': 500, 'mid': 800, 'senior': 1200, 'exec': 1600}
    node_sizes = [exp_size[G.nodes[n]['experience']] for n in G.nodes()]

    # Draw the graph
    nx.draw(G, pos, with_labels=False, node_color=node_colors, node_size=node_sizes,
            alpha=0.8, arrows=True, arrowsize=20, edge_color="gray")

    # Add labels with name and designation
    labels = {n: f"{G.nodes[n]['name']}\n({G.nodes[n]['designation']})" for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_weight='bold')

    plt.title("Organizational Hierarchy Network", fontsize=16)
    plt.axis('off')

    # Add a legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='Male', markerfacecolor='skyblue', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Female', markerfacecolor='lightpink', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Entry', markerfacecolor='gray', markersize=8),
        plt.Line2D([0], [0], marker='o', color='w', label='Mid', markerfacecolor='gray', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Senior', markerfacecolor='gray', markersize=12),
        plt.Line2D([0], [0], marker='o', color='w', label='Executive', markerfacecolor='gray', markersize=14)
    ]
    plt.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig('org_chart.png')
    plt.show()

    # Print employee data including reportees
    print("Employee Details:")
    for node in sorted(G.nodes()):
        emp = G.nodes[node]
        reportee_names = [G.nodes[r]['name'] for r in emp['reportees']]
        print(f"ID: {node}, Name: {emp['name']}, Position: {emp['designation']}, "
            f"Experience: {emp['experience']}, Gender: {emp['gender']}")
        print(f"  Reportees ({len(reportee_names)}): {', '.join(reportee_names) if reportee_names else 'None'}")
        print()
        
    return G


def analyze_org_network_centralities(G):
    """
    Calculate various centrality measures and perform gender-based analysis.
    
    Args:
        G: NetworkX DiGraph object representing the organizational structure
        
    Returns:
        tuple: (employee_dataframe, female_distances_dataframe)
    """
    # 1. Basic centrality measures
    print("Calculating centrality measures...")
    degree_centrality = nx.degree_centrality(G)
    in_degree_centrality = nx.in_degree_centrality(G)
    out_degree_centrality = nx.out_degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)
    
    # 2. Create a DataFrame with employee data including centralities
    employee_data = []
    for node in G.nodes():
        employee = {
            'id': node,
            'name': G.nodes[node]['name'],
            'gender': G.nodes[node]['gender'],
            'experience': G.nodes[node]['experience'],
            'designation': G.nodes[node]['designation'],
            'degree_centrality': degree_centrality[node],
            'in_degree_centrality': in_degree_centrality[node],
            'out_degree_centrality': out_degree_centrality[node],
            'betweenness_centrality': betweenness_centrality[node],
            'closeness_centrality': closeness_centrality[node],
            'num_reportees': len(list(G.successors(node)))
        }
        employee_data.append(employee)
    
    df = pd.DataFrame(employee_data)
    
    # Print centrality measures summary
    print("\nCentrality Measures Summary:")
    print(df[['name', 'gender', 'experience', 'designation', 
            'degree_centrality', 'betweenness_centrality', 'closeness_centrality']].sort_values(
                by='betweenness_centrality', ascending=False))
    
    # 3. Gender-based analysis
    print("\n--- Gender-Based Analysis ---")
    gender_centrality = df.groupby('gender')[['degree_centrality', 'in_degree_centrality', 
                                            'out_degree_centrality', 'betweenness_centrality', 
                                            'closeness_centrality']].mean()
    print("\nAverage Centrality by Gender:")
    print(gender_centrality)
    
    # 4. Experience-level analysis by gender
    print("\nAverage Centrality by Gender and Experience Level:")
    gender_exp_centrality = df.groupby(['gender', 'experience'])[['degree_centrality', 
                                                                'betweenness_centrality']].mean()
    print(gender_exp_centrality)
    
    # 5. Female leadership analysis
    female_leaders = df[(df['gender'] == 'female') & 
                        (df['experience'].isin(['senior', 'exec']))]
    
    print("\nFemale Leaders:")
    print(female_leaders[['name', 'designation', 'degree_centrality', 
                        'betweenness_centrality', 'closeness_centrality']])
    
    # 6. Distance analysis between female employees
    female_employees = df[df['gender'] == 'female']['id'].tolist()
    
    print("\nPath Analysis between Female Employees:")
    
    # Create a table to store distances
    female_distances = []
    
    for i, source in enumerate(female_employees):
        for target in female_employees[i+1:]:
            source_name = df[df['id'] == source]['name'].values[0]
            target_name = df[df['id'] == target]['name'].values[0]
            
            # Find shortest path
            try:
                path = nx.shortest_path(G, source=source, target=target)
                distance = len(path) - 1  # Number of hops
                female_distances.append({
                    'source': source_name,
                    'target': target_name,
                    'distance': distance,
                    'path': [df[df['id'] == node]['name'].values[0] for node in path]
                })
            except nx.NetworkXNoPath:
                # Try reverse direction if direct path doesn't exist
                try:
                    path = nx.shortest_path(G, source=target, target=source)
                    distance = len(path) - 1
                    female_distances.append({
                        'source': source_name,
                        'target': target_name,
                        'distance': distance,
                        'path': [df[df['id'] == node]['name'].values[0] for node in path]
                    })
                except nx.NetworkXNoPath:
                    female_distances.append({
                        'source': source_name,
                        'target': target_name,
                        'distance': float('inf'),
                        'path': "No path exists"
                    })
    
    # Create DataFrame for distances
    female_distances_df = pd.DataFrame(female_distances)
    print("\nDistances between Female Employees:")
    for _, row in female_distances_df.iterrows():
        if isinstance(row['path'], list):
            path_str = " -> ".join(row['path'])
        else:
            path_str = row['path']
        print(f"{row['source']} to {row['target']}: {row['distance']} hops")
        print(f"  Path: {path_str}")
    
    # 7. Calculate average distance between female employees vs. all employees
    all_distances = []
    all_nodes = list(G.nodes())
    
    # Sample a subset of pairs for efficiency (optional)
    import random
    sample_pairs = [(i, j) for i in all_nodes for j in all_nodes if i < j]
    if len(sample_pairs) > 100:  # Limit to 100 random pairs for efficiency
        sample_pairs = random.sample(sample_pairs, 100)
    
    for source, target in sample_pairs:
        try:
            distance = nx.shortest_path_length(G, source=source, target=target)
            all_distances.append(distance)
        except nx.NetworkXNoPath:
            try:
                distance = nx.shortest_path_length(G, source=target, target=source)
                all_distances.append(distance)
            except nx.NetworkXNoPath:
                pass
    
    # Calculate average distances
    female_avg_distance = female_distances_df['distance'].replace(float('inf'), None).mean()
    all_avg_distance = sum(all_distances) / len(all_distances) if all_distances else 0
    
    print(f"\nAverage distance between female employees: {female_avg_distance:.2f} hops")
    print(f"Average distance between all employees: {all_avg_distance:.2f} hops")
    
    # 8. Visualize centrality measures by gender
    plt.figure(figsize=(12, 8))
    
    metrics = ['degree_centrality', 'betweenness_centrality', 'closeness_centrality']
    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i+1)
        sns.boxplot(x='gender', y=metric, data=df)
        plt.title(f'{metric.replace("_", " ").title()} by Gender')
    
    plt.tight_layout()
    plt.savefig('centrality_by_gender.png')
    plt.show()
    
    # 9. Calculate female employee accessibility
    print("\nFemale Employee Accessibility Analysis:")
    accessibility = {}
    for female_id in female_employees:
        # Calculate how many nodes can be reached from this female employee
        reachable = set()
        for node in G.nodes():
            try:
                if nx.has_path(G, female_id, node):
                    reachable.add(node)
            except:
                pass
        
        accessibility[df[df['id'] == female_id]['name'].values[0]] = len(reachable) / len(G.nodes())
    
    for name, score in accessibility.items():
        print(f"{name}: Can reach {score:.2%} of the organization")
    
    return df, female_distances_df


def analyze_leadership_gender_equality(G, df):
    """
    Analyze gender equality in leadership positions within the organization.
    
    Args:
        G: NetworkX DiGraph object representing the organizational structure
        df: DataFrame containing employee data with centrality metrics
        
    Returns:
        DataFrame: Gender distribution by leadership level
    """
    print("\n--- Leadership Gender Equality Analysis ---")
    
    # Define leadership levels
    leadership_levels = {
        'exec': 3,  # Executive
        'senior': 2,  # Senior Management
        'mid': 1,    # Mid-level Management
        'entry': 0   # Entry level
    }
    
    # Add leadership level to the dataframe
    df['leadership_level'] = df['experience'].map(leadership_levels)
    
    # Analyze gender distribution by leadership level
    leadership_gender = pd.crosstab(df['leadership_level'], df['gender'])
    leadership_gender['total'] = leadership_gender.sum(axis=1)
    leadership_gender['female_pct'] = leadership_gender['female'] / leadership_gender['total'] * 100
    
    # Reverse sort by leadership level to show exec at the top
    leadership_gender = leadership_gender.sort_index(ascending=False)
    
    print("\nGender Distribution by Leadership Level:")
    print(leadership_gender)
    
    # Calculate gender representation gap at each level
    avg_female_pct = df[df['gender'] == 'female'].shape[0] / df.shape[0] * 100
    leadership_gender['representation_gap'] = leadership_gender['female_pct'] - avg_female_pct
    
    print(f"\nOverall female representation: {avg_female_pct:.1f}%")
    print("\nRepresentation gap by level (positive means over-representation):")
    level_names = {3: 'Executive', 2: 'Senior Management', 1: 'Mid-level Management', 0: 'Entry level'}
    for level, row in leadership_gender.iterrows():
        print(f"{level_names[level]}: {row['representation_gap']:.1f}%")
    
    # Visualize the gender distribution by leadership level
    plt.figure(figsize=(10, 6))
    leadership_gender.plot(kind='bar', y=['female', 'male'], stacked=True)
    plt.title('Gender Distribution by Leadership Level')
    plt.xlabel('Leadership Level')
    plt.ylabel('Number of Employees')
    plt.xticks(rotation=0)
    plt.legend(['Female', 'Male'])
    
    # Add percentage labels
    for i, level in enumerate(leadership_gender.index):
        plt.annotate(f"{leadership_gender.loc[level, 'female_pct']:.1f}%", 
                    xy=(i, leadership_gender.loc[level, 'female'] / 2),
                    ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig('leadership_gender_distribution.png')
    plt.show()
    
    return leadership_gender


import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import io
import base64
import networkx as nx

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def run():
    st.header("📊 Gender Diversity Analysis")
    st.markdown("""
        Analyze gender distribution across your organization to identify patterns, 
        representation gaps, and opportunities for improvement.
    """)
    
    # Create tabs for different analyses
    tabs = st.tabs([
        "Organization Gender Chart", 
        "Hierarchy Network", 
        "Leadership Analysis", 
        "Centrality Analysis"
    ])
    
    # Tab 1: Organization Gender Chart
    with tabs[0]:
        st.subheader("Organizational Gender Distribution")
        st.markdown("""
            This interactive chart shows the gender distribution across different departments in your organization.
            Color gradient represents gender ratio - blue for male-dominated, pink for female-dominated areas.
        """)
        
        # Display the gender org chart using plotly
        fig = create_gender_org_chart(width=700, height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Add explanation text
        st.markdown("""
            **How to read this chart:**
            - Each node represents a department or team
            - Color indicates gender ratio (blue = male-dominated, pink = female-dominated)
            - Hover over nodes to see exact percentages
            - Connected nodes show reporting relationships
        """)
    
    # Tab 2: Organizational Hierarchy Network
    with tabs[1]:
        st.subheader("Organizational Hierarchy Network")
        st.markdown("""
            This network visualization shows your organizational structure with gender distribution highlighted.
        """)
        
        # Create org chart (this returns a NetworkX graph)
        G = create_org_chart()
        
        # Create Plotly figure for the network graph
        pos = nx.spring_layout(G)  # You can use other layouts as needed
        
        # Create edge traces
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')
        
        # Create node traces
        node_x = []
        node_y = []
        node_colors = []
        node_text = []
        node_sizes = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_colors.append('rgb(255, 182, 193)' if G.nodes[node]['gender'] == 'female' else 'rgb(173, 216, 230)')
            node_text.append(f"{G.nodes[node]['name']}<br>{G.nodes[node]['designation']}<br>Gender: {G.nodes[node]['gender']}")
            
            # Size based on experience level
            exp_size = {'entry': 10, 'mid': 15, 'senior': 20, 'exec': 25}
            node_sizes.append(exp_size[G.nodes[node]['experience']])
        
        node_trace = go.Scatter( 
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=[G.nodes[node]['name'] for node in G.nodes()],
            textposition="bottom center",
            marker=dict(
                color=node_colors,
                size=node_sizes,
                line=dict(width=2, color='DarkSlateGrey')))
        
        # Create the figure
        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=0, l=0, r=0, t=0),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            height=600
                        ))
        
        # Add legend
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, color='rgb(255, 182, 193)'),
            name='Female'
        ))
        
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, color='rgb(173, 216, 230)'),
            name='Male'
        ))
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add employee details in an expander
        with st.expander("View Employee Details"):
            # Create a dataframe from employee data in the graph
            employees_data = []
            for node in sorted(G.nodes()):
                emp = G.nodes[node]
                reportee_names = [G.nodes[r]['name'] for r in G.successors(node)]
                employees_data.append({
                    "ID": node,
                    "Name": emp['name'],
                    "Position": emp['designation'],
                    "Gender": emp['gender'],
                    "Experience Level": emp['experience'],
                    "# Direct Reports": len(list(G.successors(node))),
                    "Reports To": G.nodes[list(G.predecessors(node))[0]]['name'] if list(G.predecessors(node)) else "None"
                })
            
            emp_df = pd.DataFrame(employees_data)
            st.dataframe(emp_df, use_container_width=True)
    
    # Tab 3: Leadership Analysis
    with tabs[2]:
        st.subheader("Leadership Gender Equality Analysis")
        st.markdown("""
            Examine gender representation across different leadership levels in your organization.
        """)
        
        # Run analysis (assumes we have employee_df from earlier)
        employee_df, _ = analyze_org_network_centralities(G)
        leadership_gender_df = analyze_leadership_gender_equality(G, employee_df)
        
        # Create Plotly figure for leadership distribution
        level_names = {3: 'Executive', 2: 'Senior Management', 1: 'Mid-level Management', 0: 'Entry level'}
        leadership_gender_df['Level'] = leadership_gender_df.index.map(level_names)
        
        fig = px.bar(leadership_gender_df, 
                     x='Level', 
                     y=['female', 'male'],
                     color_discrete_map={'female': 'rgb(255, 182, 193)', 'male': 'rgb(173, 216, 230)'},
                     title='Gender Distribution by Leadership Level',
                     labels={'value': 'Number of Employees', 'variable': 'Gender'},
                     barmode='stack')
        
        # Add percentage annotations
        for i, row in leadership_gender_df.iterrows():
            fig.add_annotation(
                x=level_names[i],
                y=row['female']/2,
                text=f"{row['female_pct']:.1f}%",
                showarrow=False,
                font=dict(color='black')
            )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display the gender representation gap data
        st.subheader("Gender Representation by Leadership Level")
        
        # Create formatted dataframe for display
        display_data = []
        overall_female_pct = employee_df[employee_df['gender'] == 'female'].shape[0] / employee_df.shape[0] * 100
        
        for level, row in leadership_gender_df.iterrows():
            display_data.append({
                "Level": level_names[level],
                "Female (%)": f"{row['female_pct']:.1f}%",
                "Male (%)": f"{100 - row['female_pct']:.1f}%",
                "Representation Gap": f"{row['representation_gap']:+.1f}%" 
            })
        
        st.dataframe(pd.DataFrame(display_data), use_container_width=True)
        
        st.info(f"Overall female representation: {overall_female_pct:.1f}%")
        
        # Recommendations
        st.subheader("Recommendations")
        
        # Generate recommendations based on the data
        levels_below_avg = [level for level, row in leadership_gender_df.iterrows() 
                          if row['representation_gap'] < -5]  # 5% threshold
        
        if levels_below_avg:
            level_names_below = [level_names[l] for l in levels_below_avg]
            st.warning(f"Consider improving gender diversity at: {', '.join(level_names_below)}")
            
            st.markdown("""
                **Suggested actions:**
                - Review promotion criteria for potential bias
                - Implement mentorship programs for underrepresented groups
                - Consider blind resume screening for hiring
                - Set diversity goals with accountability measures
            """)
        else:
            st.success("Gender representation appears relatively balanced across leadership levels.")
    
    # Tab 4: Centrality Analysis
    with tabs[3]:
        st.subheader("Network Centrality Analysis by Gender")
        st.markdown("""
            Analyze influence, connectivity, and position of employees by gender in the organizational network.
        """)
        
        # First, ensure employee_df has been created
        if 'employee_df' not in locals():
            employee_df, _ = analyze_org_network_centralities(G)
        
        # Create a custom function to generate individual boxplots
        def create_boxplot(df, y_column):
            fig = px.box(
                df, 
                x='gender', 
                y=y_column,
                color='gender',
                title=y_column.replace('_', ' ').title(),
                color_discrete_map={'female': 'rgb(255, 182, 193)', 'male': 'rgb(173, 216, 230)'},
                points="all"  # Show all points
            )
            fig.update_layout(showlegend=False)
            return fig
        
        # Create individual plots for the centrality metrics
        metrics = ['degree_centrality', 'betweenness_centrality', 'closeness_centrality']
        
        # Create 3 columns for the metrics
        cols = st.columns(3)
        
        # Display each metric in its own column
        for i, metric in enumerate(metrics):
            with cols[i]:
                st.plotly_chart(
                    create_boxplot(employee_df, metric),
                    use_container_width=True
                )
        
        # Explain what each centrality measure means
        st.markdown("""
        **Understanding centrality measures:**
        - **Degree Centrality**: Number of direct connections - shows how well-connected someone is
        - **Betweenness Centrality**: How often someone lies on the shortest path between others - shows influence as an information bridge
        - **Closeness Centrality**: How close someone is to everyone else - shows efficiency in information sharing
        """)
        
        # Add summary statistics
        st.subheader("Centrality Summary by Gender")
        
        # Calculate mean centrality by gender
        centrality_summary = employee_df.groupby('gender')[metrics].mean().reset_index()
        
        # Format the summary for display
        formatted_summary = pd.DataFrame()
        formatted_summary['Gender'] = centrality_summary['gender']
        
        for metric in metrics:
            formatted_summary[metric.replace('_', ' ').title()] = centrality_summary[metric].apply(lambda x: f"{x:.3f}")
        
        st.dataframe(formatted_summary, use_container_width=True)
        
        # Add female employee accessibility analysis
        st.subheader("Female Employee Organizational Reach")
        
        # Calculate accessibility for display
        female_employees = employee_df[employee_df['gender'] == 'female']['id'].tolist()
        accessibility = {}
        for female_id in female_employees:
            # Calculate how many nodes can be reached from this female employee
            reachable = set()
            for node in G.nodes():
                try:
                    if nx.has_path(G, female_id, node):
                        reachable.add(node)
                except:
                    pass
            
            accessibility[employee_df[employee_df['id'] == female_id]['name'].values[0]] = len(reachable) / len(G.nodes())
        
        # Create dataframe for display
        accessibility_df = pd.DataFrame([
            {"Employee": name, "Organizational Reach": score, "Percentage": f"{score:.1%}"}
            for name, score in accessibility.items()
        ]).sort_values(by="Organizational Reach", ascending=False)
        
        # Create a bar chart for organizational reach
        fig = px.bar(
            accessibility_df, 
            x='Employee', 
            y='Organizational Reach',
            color_discrete_sequence=['rgb(255, 182, 193)'],
            labels={'Organizational Reach': 'Reach (% of organization)'},
            text='Percentage'
        )
        
        fig.update_layout(
            xaxis_tickangle=-45,
            yaxis=dict(tickformat='.0%')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Final insights
        avg_reach = sum(accessibility.values()) / len(accessibility)
        st.info(f"On average, female employees can reach {avg_reach:.1%} of the organization through direct reporting lines.")
