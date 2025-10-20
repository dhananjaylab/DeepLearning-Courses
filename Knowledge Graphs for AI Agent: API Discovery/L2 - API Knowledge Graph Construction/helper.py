import os
import re

import networkx as nx
import numpy as np
import pandas as pd
import tqdm
from dotenv import find_dotenv, load_dotenv
from faiss import IndexFlat, IndexFlatL2
from langchain.embeddings import OpenAIEmbeddings
from rdflib import Dataset, Graph, Literal
from rdflib.plugins.sparql.processor import prepareQuery


data_prs = [
    {
        "PurchaseRequisition": "4000000001",
        "PurchaseReqnItem": [
            {
                "Material": "ruler",
                "RequestedQuantity": 1,
                "PurchasingGroup": "005",
                "PurchasingOrganization": "3000",
                "PurchaseRequisition": "4000000001",
                "PurchaseRequisitionItem": "00010",
            }
        ],
    },
    {
        "PurchaseRequisition": "4000000002",
        "PurchaseReqnItem": [
            {
                "Material": "pencil",
                "RequestedQuantity": 5,
                "PurchasingGroup": "002",
                "PurchasingOrganization": "3000",
                "PurchaseRequisition": "4000000002",
                "PurchaseRequisitionItem": "00010",
            },
            {
                "Material": "pen",
                "RequestedQuantity": 3,
                "PurchasingGroup": "002",
                "PurchasingOrganization": "3000",
                "PurchaseRequisition": "4000000002",
                "PurchaseRequisitionItem": "00020",
            },
        ],
    },
]

data_pos = [
    {
        "PurchaseOrder": "4500000001",
        "PurchasingOrganization": "3000",
        "PurchasingGroup": "005",
        "PurchasingProcessingStatus": "01",
        "PurchaseOrderItem": [
            {
                "PurchaseOrder": "4500000001",
                "PurchaseOrderItem": "00010",
                "OrderQuantity": 1,
                "Material": "ruler",
                "PurchaseRequisition": "4000000001",
                "PurchaseRequisitionItem": "00010",
            }
        ],
    },
    {
        "PurchaseOrder": "4500000002",
        "PurchasingOrganization": "3000",
        "PurchasingGroup": "002",
        "PurchasingProcessingStatus": "02",
        "PurchaseOrderItem": [
            {
                "PurchaseOrder": "4500000002",
                "PurchaseOrderItem": "00010",
                "OrderQuantity": 5,
                "Material": "pencil",
                "PurchaseRequisition": "4000000002",
                "PurchaseRequisitionItem": "00010",
            }
        ],
    },
    {
        "PurchaseOrder": "4500000003",
        "PurchasingOrganization": "3000",
        "PurchasingGroup": "002",
        "PurchasingProcessingStatus": "02",
        "PurchaseOrderItem": [
            {
                "PurchaseOrder": "4500000003",
                "PurchaseOrderItem": "00010",
                "OrderQuantity": 3,
                "Material": "pen",
                "PurchaseRequisition": "4000000002",
                "PurchaseRequisitionItem": "00020",
            }
        ],
    },
]

data = {"PURCHASEORDER": data_pos, "PURCHASEREQN": data_prs}

# these expect to find a .env file at the directory above the lesson.                                                                                                                     # the format for that file is (without the comment)                                                                                                                                       #API_KEYNAME=AStringThatIsTheLongAPIKeyFromSomeService
def load_env():
    _ = load_dotenv(find_dotenv())


def get_openai_api_key():
    load_env()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    return openai_api_key


def parameterize_sparql(query: str, parameters: dict) -> str:
    for key, value in parameters.items():
        query = re.sub(f"var:::{key}", value, query)
    return query


def query_index(
    index: IndexFlat,
    entity_set_uris: list[str],
    embedding_model: OpenAIEmbeddings,
    query: str,
    top: int = 5,
) -> list[str]:
    # embedd the user query
    x_query = np.array([embedding_model.embed_query(query)])

    # find the indices of the top k most similar embeddings
    _, indices = index.search(x_query, top)

    # return entity_sets that correspond to the indices
    return [entity_set_uris[i] for i in indices[0]]


def get_process_dependencies(
    entity_set_uris: list[str], graph: Dataset
) -> list[tuple[str, str, str, str]]:
    entity_set_uris = " ".join([f"<{uri}>" for uri in entity_set_uris])
    get_process_dependencies = """
    PREFIX pr: <http://example.org/process#>
    PREFIX odata: <http://example.org/odata#>
    SELECT DISTINCT ?entitySetA ?entitySetB ?nameA ?nameB
    WHERE {
        {
        VALUES ?entitySetA { var:::entity_set_uris }
        ?activityA  pr:entitySet ?entitySetA ;
                    pr:hasNext ?activityB . 
        
        ?activityB pr:entitySet ?entitySetB .
        ?entitySetA odata:name ?nameA .
        ?entitySetB odata:name ?nameB .
        } 
        UNION {
        VALUES ?entitySetB { var:::entity_set_uris }
        ?activityA  pr:entitySet ?entitySetA ;
                    pr:hasNext ?activityB . 
        
        ?activityB pr:entitySet ?entitySetB .
        ?entitySetA odata:name ?nameA .
        ?entitySetB odata:name ?nameB .
        }
    }
    """
    return [
        (str(row.entitySetA), str(row.entitySetB), str(row.nameA), str(row.nameB))
        for row in graph.query(
            parameterize_sparql(
                query=get_process_dependencies,
                parameters={"entity_set_uris": entity_set_uris},
            )
        )
    ]


def discover_apis_and_process(
    query: str,
    graph: Dataset,
    index: IndexFlat,
    entity_set_uris: list[str],
    embedding_model: OpenAIEmbeddings,
) -> dict:
    # embedding based retrieval of the top 5 entity sets based on the query
    retrieved_entity_set_uris = query_index(
        index=index,
        entity_set_uris=entity_set_uris,
        embedding_model=embedding_model,
        query=query,
        top=5,
    )

    # get the process dependencies for the retrieved entity sets
    dependencies = get_process_dependencies(
        entity_set_uris=retrieved_entity_set_uris, graph=graph
    )

    # process information can contain new entity sets that are not in the retrieved entity sets
    # so we merge the retrieved entity sets with those.
    merged_entity_sets = set(retrieved_entity_set_uris)

    # create a list of process information strings
    process_information = []
    for dependency in dependencies:
        merged_entity_sets.add(dependency[0])
        merged_entity_sets.add(dependency[1])
        process_information.append(f"{dependency[3]} depends on {dependency[2]}")

    return {
        "entity_sets": merged_entity_sets,
        "process_information": process_information,
    }


def apply_filter(entities: list[dict], filter_string=None):
    if not filter_string:
        return entities
    # Split into [cond, op, cond, op, cond, ...]
    tokens = re.split(r"\s+(and|or)\s+", filter_string)

    # Parse a single condition like "Field eq 'Value'"
    cond_re = re.compile(r"^(\w+)\s+(eq|ne|gt|lt|ge|le)\s+'([^']*)'$")

    def eval_cond(entity, cond):
        m = cond_re.match(cond.strip())
        if not m:
            raise ValueError(f"Invalid condition: {cond!r}")
        field, op, raw_val = m.groups()

        # Grab the value from the entity (or None if missing)
        left = entity.get(field, None)

        # Cast filter value to the type of the left value if possible
        if isinstance(left, (int, float)):
            try:
                val = type(left)(raw_val)
            except ValueError:
                # Fallback to use string
                val = raw_val
        else:
            val = raw_val

        if op == "eq":
            return left == val
        if op == "ne":
            return left != val
        if op == "gt":
            return left > val
        if op == "lt":
            return left < val
        if op == "ge":
            return left >= val
        if op == "le":
            return left <= val

        # Fallback if operator is not recognized
        return False

    def filter_record(entity: dict) -> bool:
        # evaluate the first condition
        result = eval_cond(entity, tokens[0])

        # add subsequent conditions
        for op, cond in zip(tokens[1::2], tokens[2::2]):
            if op == "and":
                result = result and eval_cond(entity, cond)
            else:
                result = result or eval_cond(entity, cond)
        return result

    return [apply_selects(record) for record in entities if filter_record(record)]


def apply_selects(
    entities: list[dict], selects_string: str | None = None
) -> list[dict]:
    if not selects_string:
        return entities
    selects = [select.strip() for select in selects_string.split(",")]
    return [
        {field: entity[field] for field in selects if field in entity}
        for entity in entities
    ]


def get_records(
    entities: dict[str, list[dict]],
    filter_string: str | None = None,
    selects_string: str | None = None,
) -> list[dict]:
    return apply_selects(apply_filter(entities, filter_string), selects_string)


def get_data_mock(
    data: dict[str, list[dict]],
    service_name: str,
    entity_set: str = None,
    filter_string: str | None = None,
    selects_string: str | None = None,
) -> dict:
    odata_query = f"/sap/opu/odata/sap/{service_name}/{entity_set}"
    if filter_string:
        odata_query += f"?$filter={filter_string}"
    if selects_string:
        odata_query += f"&$select={selects_string}"

    print(f"Calling: GET {odata_query}")

    return get_records(
        entities=data[entity_set],
        filter_string=filter_string,
        selects_string=selects_string,
    )


def get_key(data: dict, key: str) -> str:
    """Pick out key from dictionary, case insensitive."""
    key_lower = key.lower()
    for k in data.keys():
        if k.lower() == key_lower:
            return k
    return key


def post_data_mock(
    data: dict[str, list[dict]],
    service_name: str,
    entity_set: str | None = None,
    payload: dict | None = None,
) -> dict:
    odata_query = f"/sap/opu/odata/sap/{service_name}/{entity_set}"

    print(f"Calling: POST {odata_query} with payload {payload}")

    record = payload
    new_id = str(len(data[entity_set]) + 1).zfill(8)
    if entity_set == "PURCHASEREQN":
        record[get_key(record, "PurchaseRequisition")] = "40" + new_id
        for n, item in enumerate(record.get(get_key(record, "PurchaseReqnItem"), [])):
            item[get_key(item, "PurchaseRequisition")] = "40" + new_id
            item[get_key(item, "PurchaseRequisitionItem")] = str(n + 1).zfill(4) + "0"
    elif entity_set == "PURCHASEORDER":
        record[get_key(record, "PurchaseOrder")] = "45" + new_id
        for n, item in enumerate(record.get("PurchaseOrderItem", [])):
            item[get_key(item, "PurchaseOrder")] = "45" + new_id
            item[get_key(item, "PurchaseOrderItem")] = str(n + 1).zfill(4) + "0"

    data[entity_set].append(record)

    return record


def display_prs(data_prs: dict) -> pd.DataFrame:
    rows = []
    for req in data_prs:
        for item in req["PurchaseReqnItem"]:
            rows.append(
                {
                    "Purchase Requisition": req["PurchaseRequisition"],
                    "Item": item.get("PurchaseRequisitionItem", None),
                    "Material": item.get("Material", None),
                    "Requested Quantity": item.get("RequestedQuantity", None),
                    "Purchasing Group": item.get("PurchasingGroup", None),
                    "Purchasing Organization": item.get("PurchasingOrganization", None),
                }
            )

    return pd.DataFrame(rows)


def display_pos(data_pos: dict) -> pd.DataFrame:
    rows = []
    for po in data_pos:
        for item in po["PurchaseOrderItem"]:
            rows.append(
                {
                    "Purchase Order": po["PurchaseOrder"],
                    "Purchasing Group": po.get("PurchasingGroup", None),
                    "Purchasing Organization": po.get("PurchasingOrganization", None),
                    "Processing Status": po.get("PurchasingProcessingStatus", None),
                    "Item": item.get("PurchaseOrderItem", None),
                    "Material": item.get("Material", None),
                    "Order Quantity": item.get("OrderQuantity", None),
                    "Linked Requisition": item.get("PurchaseRequisition", None),
                    "Linked Requisition Item": item.get(
                        "PurchaseRequisitionItem", None
                    ),
                }
            )

    return pd.DataFrame(rows)


def fetch_navigations(graph, entity_set_uri: str) -> list[dict]:
    q_navigations = """
    PREFIX odata: <http://example.org/odata#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

    SELECT DISTINCT
        ?entity_set_uri ?navigation_uri ?name ?target_entity_set_uri ?target_entity_set_name ?cardinality

    WHERE {

        BIND(<var:::entity_set_uri> as ?entity_set_uri)
        ?entity_set_uri rdf:type odata:EntitySet.
        ?entity_set_uri odata:entityType ?entityType.
        ?navigation_uri odata:navigateFrom ?entityType .
        ?navigation_uri odata:name ?name .
        ?navigation_uri odata:navigateTo ?navigationEntityType .

        ?navigation_uri odata:multiplicity ?cardinality .

        ?target_entity_set_uri odata:entityType ?navigationEntityType .
        ?target_entity_set_uri a odata:EntitySet.
        ?target_entity_set_uri odata:name ?target_entity_set_name.

    }
    GROUP BY
    ?entity_set_uri ?navigation_uri ?name ?target_entity_set_uri ?target_entity_set_name ?cardinality
    """

    navigations = []
    for row in graph.query(
        parameterize_sparql(
            query=q_navigations,
            parameters={"entity_set_uri": entity_set_uri}
        )
    ):
        navigations.append(
            {
                "target_entity_set": str(row.target_entity_set_name),
                "cardinality": "one_to_many"
                if str(row.cardinality) == "*"
                else "one_to_one",
            }
        )

    return navigations

def fetch_properties(graph, entity_set_uri: str) -> dict:
    # query all properties of the entity set
    q_entity_set_properties = """
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
    PREFIX odata: <http://example.org/odata#>
    SELECT DISTINCT
        ?service_uri ?service ?entity_set_uri ?entity_set 
        ?property_name ?property_type ?max_length ?is_key
    WHERE {
        BIND(<var:::entity_set_uri> as ?entity_set_uri)
        ?entity_set_uri rdf:type odata:EntitySet.
        ?entity_set_uri odata:name ?entity_set.
        ?entity_set_uri odata:entityType ?entity_type_uri.
        ?entity_type_uri odata:name ?entity_type.
        ?entity_type_uri odata:service ?service_uri.
        ?service_uri rdf:type odata:Service.
        ?service_uri odata:name ?service.

        # Properties on the EntitySet
        {
            ?entity_type_uri odata:property ?property_uri.
            ?property_uri odata:name ?property_name.
            ?property_uri odata:selectProperty true
        }
        OPTIONAL{
            ?property_uri odata:maxLength ?max_length .
        }
        OPTIONAL{
            ?property_uri odata:odataKey ?is_key .
        }

        ?property_uri odata:type ?property_type.
    }
    """

    properties = {}
    for row in graph.query(
        parameterize_sparql(
            query=q_entity_set_properties, parameters={"entity_set_uri": entity_set_uri}
        )
    ):
        properties[str(row.property_name)] = {
            "type": str(row.property_type),
            "is_key": str(row.is_key),
            "max_length": int(float((row.max_length))) if row.max_length else None,
            "fixed_values": [],
        }

    # query fixed values for all properties of the entity set
    query_fixed_values = """
    PREFIX odata: <http://example.org/odata#>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
    SELECT DISTINCT ?property_name ?key ?value
    WHERE {
        BIND(<var:::entity_set_uri> as ?entity_set_uri)
        ?entity_set_uri a odata:EntitySet.
        ?entity_set_uri odata:entityType ?entityType .
        ?entityType odata:property ?property .
        ?property odata:name ?property_name.
        OPTIONAL {?property odata:valueHelp [ odata:key ?key ; odata:value ?value ] .}
    }
    """

    for row in graph.query(
        parameterize_sparql(
            query=query_fixed_values,
            parameters={"entity_set_uri": entity_set_uri}
        )
    ):
        if row.key is not None and row.value is not None:
            properties[str(row.property_name)]["fixed_values"].append(
                {"value": str(row.key), "description": str(row.value)}
            )

    return properties

def fetch_entity_specification(graph, entity_set_uri: str) -> dict:
    properties = fetch_properties(graph, entity_set_uri)
    navigations = fetch_navigations(graph, entity_set_uri)
    
    m1 = re.search(r"/Service/([^/]+)/", entity_set_uri)
    service = m1.group(1) if m1 else None
    m2 = re.search(r"/EntitySet/([^/]+)$", entity_set_uri)
    entity_set = m2.group(1) if m2 else None

    return {
        "service": service,
        "entity_set": entity_set,
        "properties": properties,
        "navigations": navigations,
    }


# Transform function
def transform(df: pd.DataFrame, construct_query: str, first: bool = False) -> Graph:
    """Transform Pandas DataFrame to RDFLib Graph given a SPARQL Construct Query."""

    # Setup query graph and parse query
    query_graph = Graph()
    query = prepareQuery(construct_query)
    # Transfer namespaces from query to graph for serialization
    for prefix, uri in query.prologue.namespace_manager.namespaces():
        query_graph.bind(prefix, uri)

    graph = Graph()
    # Get headers
    invalid_pattern = re.compile(r"[^\w_]+")
    headers = dict((k, invalid_pattern.sub("_", k)) for k in df.columns)
    for _, row in df.iterrows():
        binding = dict(
            (headers[k], Literal(row[k])) for k in df.columns if len(str(row[k])) > 0
        )
        results = query_graph.query(query, initBindings=binding)
        for result in results:
            graph.add(result)
        if first:
            break
    return graph


def rdf_to_nx(graph: Graph) -> nx.Graph:
    """Convert RDFLib Graph to NetworkX Graph."""
    nx_graph = nx.Graph()
    edges_query = """
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    SELECT ?node1 ?node2
    WHERE {
        ?node1 ?edge ?node2 .
        FILTER(?edge != rdf:type)
        FILTER(!isLiteral(?node2))
    }
    """
    edges = graph.query(edges_query)
    edges_list = [(str(row.node1), str(row.node2)) for row in edges]

    nx_graph = nx.Graph()
    nx_graph.add_edges_from(edges_list)
    return nx_graph
