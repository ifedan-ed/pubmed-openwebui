"""
title: Smart PubMed Research Assistant
author: Research Assistant
version: 6.0.0
date: 2025-01-01
license: MIT
description: Intelligent PubMed research assistant. Uses PubMed's own Automatic Term Mapping. Fetches abstracts for AI synthesis. Outputs Vancouver-style numbered references. RIS export for Zotero. Works with or without NCBI API key.
"""

import requests
import re
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional, Tuple
from pydantic import BaseModel, Field


class Tools:

    class Valves(BaseModel):
        ncbi_api_key: str = Field(
            default="",
            description="Optional: NCBI API key for faster searches (get free at https://www.ncbi.nlm.nih.gov/account/settings/). Leave empty to work without it."
        )

    def __init__(self):
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.valves = self.Valves()
        self._cache = {}
        self._last_results = []
        self._last_query = ""

    def _api_params(self, params: dict) -> dict:
        """Add API key only if configured"""
        if self.valves.ncbi_api_key and self.valves.ncbi_api_key.strip():
            params["api_key"] = self.valves.ncbi_api_key.strip()
        return params

    # ================================================================
    # MAIN SEARCH
    # ================================================================

    def search_pubmed(
        self,
        query: str = Field(
            ...,
            description="Your research question in plain English. Examples: 'current guidelines for management of gastric reflux in children', 'low back pain treatment', 'ECMO outcomes in neonates'",
        ),
        max_results: int = Field(
            10,
            description="How many articles (1-200). More = slower but more comprehensive.",
        ),
        include_abstracts: bool = Field(
            True,
            description="Include abstracts for AI synthesis. False = faster metadata-only search.",
        ),
    ) -> str:
        """
        Intelligent PubMed search with abstract retrieval.
        Returns Vancouver-style numbered references.
        AI can read abstracts to synthesize and answer questions.
        """
        try:
            max_results = self._safe_int(max_results, 10, 1, 200)
            query = str(query).strip()
            if not query:
                return "Please ask me a research question."

            if isinstance(include_abstracts, str):
                include_abstracts = include_abstracts.lower() not in ("false", "no", "0")
            elif not isinstance(include_abstracts, bool):
                include_abstracts = True

            # PHASE 1: Let PubMed understand the query
            analysis = self._analyze_via_pubmed(query)

            # PHASE 2: Detect query type
            query_type = self._detect_query_type(query)

            # PHASE 3: Search iteratively
            all_articles, search_log = self._iterative_search(
                query, analysis, query_type, max_results
            )

            # PHASE 4: Fetch abstracts
            if include_abstracts and all_articles:
                pmids = [a["pmid"] for a in all_articles]
                abstracts = self._fetch_abstracts(pmids)
                for article in all_articles:
                    article["abstract"] = abstracts.get(article["pmid"], "")

            # PHASE 5: Score and rank
            scored = self._score_relevance(all_articles, query, query_type)
            top = scored[:max_results]

            # PHASE 6: Assign reference numbers
            for i, article in enumerate(top):
                article["ref_number"] = i + 1

            # PHASE 7: Store
            self._last_results = top
            self._last_query = query

            # PHASE 8: Format
            if not top:
                return self._format_no_results(query, analysis, search_log)

            return self._format_results(
                query, analysis, query_type, search_log,
                top, len(all_articles), include_abstracts
            )

        except Exception as e:
            return self._error_msg(str(e))

    # ================================================================
    # GET RESULTS IN DIFFERENT FORMATS
    # ================================================================

    def get_results(
        self,
        format: str = Field(
            "list",
            description="'list' (Vancouver references), 'ris' (Zotero export), 'summary' (AI synthesis), 'abstracts' (all abstracts), 'detailed' (full metadata)",
        ),
    ) -> str:
        """
        Get last search results in different formats.
        References use Vancouver numbered style throughout.
        """
        try:
            fmt = str(format).strip().lower() if format else "list"
            if not self._last_results:
                return "No stored results. Run `search_pubmed` first."

            if fmt == "ris":
                return self._export_ris()
            elif fmt == "summary":
                return self._synthesize()
            elif fmt == "abstracts":
                return self._format_abstracts_only()
            elif fmt == "detailed":
                return self._format_detailed()
            else:
                return self._format_vancouver_list()
        except Exception as e:
            return self._error_msg(str(e))

    # ================================================================
    # PICO SEARCH
    # ================================================================

    def pico_search(
        self,
        population: str = Field(..., description="Who? e.g. 'children under 5 in Africa'"),
        intervention: str = Field("", description="What? e.g. 'proton pump inhibitors'"),
        comparison: str = Field("", description="Versus? e.g. 'lifestyle modification'"),
        outcome: str = Field("", description="Result? e.g. 'symptom resolution'"),
        max_results: int = Field(15, description="How many articles (1-200)"),
    ) -> str:
        """PICO framework search with abstracts and Vancouver references."""
        try:
            max_results = self._safe_int(max_results, 15, 1, 200)

            pico = {}
            for label, val in [("Population", population), ("Intervention", intervention),
                               ("Comparison", comparison), ("Outcome", outcome)]:
                val = str(val).strip() if val else ""
                if val:
                    pico[label] = val

            if not pico:
                return "Please provide at least a Population."

            pico_analysis = {}
            for comp, text in pico.items():
                pico_analysis[comp] = self._analyze_via_pubmed(text)

            all_articles, search_log = self._pico_iterative_search(
                pico, pico_analysis, max_results
            )

            # Fetch abstracts
            if all_articles:
                pmids = [a["pmid"] for a in all_articles]
                abstracts = self._fetch_abstracts(pmids)
                for a in all_articles:
                    a["abstract"] = abstracts.get(a["pmid"], "")

            combined = " ".join(pico.values())
            scored = self._score_relevance(
                all_articles, combined, self._detect_query_type(combined)
            )
            top = scored[:max_results]

            # Assign reference numbers
            for i, a in enumerate(top):
                a["ref_number"] = i + 1

            self._last_results = top
            self._last_query = "PICO: " + "; ".join(
                k + "=" + v for k, v in pico.items()
            )

            # Format
            md = "# 🔬 PICO Search Results\n\n"
            md += "## Framework\n\n"
            md += "| Component | Input | PubMed Mapped To |\n"
            md += "|-----------|-------|------------------|\n"
            for comp, text in pico.items():
                a = pico_analysis[comp]
                mapped = ", ".join(a.get("mesh_found", [])[:3]) or text
                md += "| **" + comp + "** | " + text + " | " + mapped + " |\n"
            md += "\n"

            for s in search_log:
                icon = "✅" if s["found"] > 0 else "⭕"
                md += icon + " **" + s["name"] + "** → " + str(s["found"]) + "  \n"
            md += "\n"

            if top:
                with_abs = sum(1 for a in top if a.get("abstract"))
                md += "## Results (" + str(len(top)) + " articles"
                if with_abs:
                    md += ", " + str(with_abs) + " with abstracts"
                md += ")\n\n"
                md += self._format_article_list(top, show_abstracts=True)
                md += "\n## References (Vancouver Style)\n\n"
                md += self._build_vancouver_list(top)
            else:
                md += "**No results.** Try broader terms.\n"

            md += self._format_next_steps()
            return md

        except Exception as e:
            return self._error_msg(str(e))

    # ================================================================
    # MESH FINDER
    # ================================================================

    def find_mesh(
        self,
        topic: str = Field(..., description="Any medical topic"),
    ) -> str:
        """Find MeSH terms via PubMed's own term mapping."""
        try:
            analysis = self._analyze_via_pubmed(str(topic).strip())
            md = "# 🏷️ MeSH: " + topic + "\n\n"
            if analysis["mesh_found"]:
                md += "| MeSH Term | Syntax |\n|---|---|\n"
                for t in analysis["mesh_found"]:
                    md += "| " + t + " | `\"" + t + "\"[MeSH]` |\n"
                md += "\n"
            if analysis["query_translation"]:
                md += "**PubMed translation:**\n```\n" + analysis["query_translation"] + "\n```\n"
            return md
        except Exception as e:
            return self._error_msg(str(e))

    # ================================================================
    # CORE: PUBMED QUERY ANALYSIS
    # ================================================================

    def _analyze_via_pubmed(self, query: str) -> Dict:
        """Let PubMed itself understand the query — no word splitting"""

        cache_key = query.lower().strip()
        if cache_key in self._cache:
            return self._cache[cache_key]

        result = {
            "original": query,
            "query_translation": "",
            "mesh_found": [],
            "result_count": 0,
        }

        try:
            resp = requests.get(
                self.base_url + "/esearch.fcgi",
                params=self._api_params({
                    "db": "pubmed", "term": query,
                    "retmode": "json", "retmax": "0",
                }),
                timeout=15,
            )
            if resp.status_code == 200:
                es = resp.json().get("esearchresult", {})
                result["result_count"] = int(es.get("count", 0))
                result["query_translation"] = es.get("querytranslation", "")
                if result["query_translation"]:
                    mesh = re.findall(
                        r'"([^"]+)"\[MeSH Terms\]',
                        result["query_translation"]
                    )
                    result["mesh_found"] = list(dict.fromkeys(mesh))

            # Also try full phrase as MeSH
            phrase_query = '"' + query + '"[MeSH Terms]'
            resp2 = requests.get(
                self.base_url + "/esearch.fcgi",
                params=self._api_params({
                    "db": "pubmed", "term": phrase_query,
                    "retmode": "json", "retmax": "0",
                }),
                timeout=10,
            )
            if resp2.status_code == 200:
                es2 = resp2.json().get("esearchresult", {})
                trans2 = es2.get("querytranslation", "")
                if int(es2.get("count", 0)) > 0 and trans2:
                    for t in re.findall(r'"([^"]+)"\[MeSH Terms\]', trans2):
                        if t not in result["mesh_found"]:
                            result["mesh_found"].append(t)

        except Exception:
            pass

        self._cache[cache_key] = result
        return result

    # ================================================================
    # QUERY TYPE DETECTION
    # ================================================================

    def _detect_query_type(self, query: str) -> str:
        q = query.lower()
        type_map = {
            "guidelines": [
                "guideline", "guidelines", "protocol", "recommendation",
                "consensus", "management of"
            ],
            "systematic_review": ["systematic review", "meta-analysis"],
            "outcomes": ["outcome", "outcomes", "effectiveness", "efficacy"],
            "epidemiology": ["prevalence", "incidence", "epidemiology"],
            "diagnosis": ["diagnosis", "diagnostic", "screening"],
            "treatment": ["treatment", "therapy", "drug", "medication"],
            "risk_factors": ["risk factor", "cause", "etiology"],
        }
        for qtype, words in type_map.items():
            if any(w in q for w in words):
                return qtype
        return "general"

    def _get_type_filter(self, qt: str) -> str:
        filters = {
            "guidelines": " AND (\"Practice Guideline\"[PT] OR \"Guideline\"[PT] OR guideline[ti])",
            "systematic_review": " AND (\"Systematic Review\"[PT] OR \"Meta-Analysis\"[PT])",
            "outcomes": " AND (\"Clinical Trial\"[PT] OR \"Comparative Study\"[PT])",
            "diagnosis": " AND (diagnosis[ti] OR diagnostic[ti])",
            "epidemiology": " AND (prevalence[ti] OR incidence[ti] OR epidemiology[sh])",
            "treatment": " AND (\"Clinical Trial\"[PT] OR \"Randomized Controlled Trial\"[PT])",
        }
        return filters.get(qt, "")

    # ================================================================
    # SEARCH STRATEGIES
    # ================================================================

    def _build_strategies(self, query, analysis, query_type):
        strategies = []
        mesh = analysis.get("mesh_found", [])
        count = analysis.get("result_count", 0)
        tf = self._get_type_filter(query_type)

        # Strategy 1: Let PubMed auto-map
        if count > 0:
            strategies.append(("PubMed Auto-Mapping", query))

        # Strategy 2: MeSH + type filter
        if mesh and tf:
            mesh_terms = ['"' + t + '"[MeSH]' for t in mesh[:4]]
            mq = " AND ".join(mesh_terms)
            strategies.append((
                "MeSH + " + query_type + " filter",
                mq + tf
            ))

        # Strategy 3: MeSH combined
        if mesh:
            mesh_terms = ['"' + t + '"[MeSH]' for t in mesh[:4]]
            strategies.append(("MeSH Combined", " AND ".join(mesh_terms)))

        # Strategy 4: Core MeSH (top 2)
        if len(mesh) >= 2:
            q = '"' + mesh[0] + '"[MeSH] AND "' + mesh[1] + '"[MeSH]'
            strategies.append(("Core MeSH", q))

        # Strategy 5: Primary MeSH + filter
        if mesh and tf:
            q = '"' + mesh[0] + '"[MeSH]' + tf
            strategies.append(("Primary MeSH + " + query_type, q))

        # Strategy 6: Title/Abstract
        strategies.append(("Title/Abstract", "(" + query + ")[tiab]"))

        # Strategy 7: All fields
        strategies.append(("All Fields", query))

        return strategies

    def _iterative_search(self, query, analysis, query_type, max_results):
        strategies = self._build_strategies(query, analysis, query_type)
        return self._run_strategies(strategies, max_results)

    def _pico_iterative_search(self, pico, pico_analysis, max_results):
        strategies = []

        # Build per-component queries
        comp_parts = []
        for comp, analysis in pico_analysis.items():
            mesh = analysis.get("mesh_found", [])
            if mesh:
                if len(mesh) > 1:
                    mesh_terms = ['"' + t + '"[MeSH]' for t in mesh[:3]]
                    joined = " OR ".join(mesh_terms)
                    comp_parts.append("(" + joined + ")")
                else:
                    comp_parts.append('"' + mesh[0] + '"[MeSH]')
            else:
                comp_parts.append("(" + pico[comp] + ")")

        # Strategy 1: Full PICO
        if len(comp_parts) >= 2:
            strategies.append(("Full PICO", " AND ".join(comp_parts)))

        # Strategy 2: Natural language
        combined = " ".join(pico.values())
        strategies.append(("Natural Language", combined))

        # Strategy 3: Component pairs
        for pair_name, keys in [("P+I", ["Population", "Intervention"]),
                                 ("P+O", ["Population", "Outcome"])]:
            parts = []
            for k in keys:
                if k in pico_analysis:
                    mesh = pico_analysis[k].get("mesh_found", [])
                    if mesh:
                        parts.append('"' + mesh[0] + '"[MeSH]')
                    elif k in pico:
                        parts.append(pico[k])
            if len(parts) == 2:
                strategies.append((pair_name, " AND ".join(parts)))

        # Strategy 4: Broad
        strategies.append(("Broad", combined))

        return self._run_strategies(strategies, max_results)

    def _run_strategies(self, strategies, max_results):
        all_articles = []
        seen = set()
        log = []

        for name, sq in strategies:
            if not sq:
                continue
            fetch_count = min(max_results * 2, 200)
            results = self._run_search(sq, fetch_count)
            log.append({"name": name, "query": sq, "found": len(results)})

            for a in results:
                if a["pmid"] not in seen:
                    seen.add(a["pmid"])
                    a["found_via"] = name
                    all_articles.append(a)

            if len(all_articles) >= max_results * 3 and len(log) >= 3:
                break

        return all_articles, log

    # ================================================================
    # ABSTRACT FETCHING
    # ================================================================

    def _fetch_abstracts(self, pmids: List[str]) -> Dict[str, str]:
        """Fetch full abstracts. Batched. Works with or without API key."""

        abstracts = {}
        if not pmids:
            return abstracts

        batch_size = 25

        for i in range(0, len(pmids), batch_size):
            batch = pmids[i:i + batch_size]

            try:
                resp = requests.get(
                    self.base_url + "/efetch.fcgi",
                    params=self._api_params({
                        "db": "pubmed",
                        "id": ",".join(batch),
                        "rettype": "xml",
                        "retmode": "xml",
                    }),
                    timeout=30,
                )

                if resp.status_code != 200:
                    continue

                try:
                    root = ET.fromstring(resp.content)
                except ET.ParseError:
                    self._regex_extract(batch, abstracts, resp.text)
                    continue

                for art_elem in root.findall(".//PubmedArticle"):
                    pmid_elem = art_elem.find(".//PMID")
                    if pmid_elem is None:
                        continue
                    pmid = pmid_elem.text

                    # Abstract
                    parts = []
                    abs_elem = art_elem.find(".//Abstract")
                    if abs_elem is not None:
                        for txt in abs_elem.findall("AbstractText"):
                            label = txt.get("Label", "")
                            text = self._elem_text(txt)
                            if text:
                                if label:
                                    parts.append("**" + label + ":** " + text)
                                else:
                                    parts.append(text)

                    if parts:
                        full = "\n\n".join(parts)

                        # MeSH headings
                        mesh_list = []
                        for m in art_elem.findall(".//MeshHeading/DescriptorName"):
                            if m.text:
                                mesh_list.append(m.text)
                        if mesh_list:
                            full += "\n\n**MeSH:** " + ", ".join(mesh_list[:10])

                        # Keywords
                        kw_list = []
                        for k in art_elem.findall(".//Keyword"):
                            if k.text:
                                kw_list.append(k.text)
                        if kw_list:
                            full += "\n**Keywords:** " + ", ".join(kw_list[:10])

                        abstracts[pmid] = full

            except Exception:
                continue

        return abstracts

    def _elem_text(self, elem) -> str:
        """Get all text from XML element including children"""
        parts = []
        if elem.text:
            parts.append(elem.text)
        for child in elem:
            if child.text:
                parts.append(child.text)
            if child.tail:
                parts.append(child.tail)
        return " ".join(parts).strip()

    def _regex_extract(self, pmids, abstracts, xml_text):
        """Fallback regex abstract extraction"""
        for pmid in pmids:
            if pmid in abstracts:
                continue
            pattern = (
                r"<PMID[^>]*>" + re.escape(pmid) + r"</PMID>"
                r".*?<Abstract>(.*?)</Abstract>"
            )
            match = re.search(pattern, xml_text, re.DOTALL)
            if match:
                text = re.sub(r"<[^>]+>", " ", match.group(1))
                text = re.sub(r"\s+", " ", text).strip()
                if text:
                    abstracts[pmid] = text

    # ================================================================
    # RELEVANCE SCORING
    # ================================================================

    def _score_relevance(self, articles, query, query_type):
        query_words = set(re.findall(r"[a-z]{3,}", query.lower()))
        query_words -= {
            "the", "and", "for", "with", "from", "that", "this", "are",
            "was", "were", "been", "have", "has", "how", "what", "which",
            "current", "recent", "new", "using", "based"
        }

        type_words = {
            "guidelines": ["guideline", "guidelines", "recommendation", "consensus", "management"],
            "systematic_review": ["systematic", "review", "meta-analysis"],
            "outcomes": ["outcome", "outcomes", "effectiveness", "efficacy"],
            "diagnosis": ["diagnosis", "diagnostic", "screening"],
            "treatment": ["treatment", "therapy", "therapeutic"],
            "epidemiology": ["prevalence", "incidence", "epidemiology"],
        }

        for a in articles:
            score = 0
            tl = a.get("title", "").lower()
            ab = a.get("abstract", "").lower()

            title_words = set(re.findall(r"[a-z]{3,}", tl))
            score += len(query_words & title_words) * 5

            if ab:
                ab_words = set(re.findall(r"[a-z]{3,}", ab))
                score += min(15, len(query_words & ab_words) * 2)
                score += 3

            for w in type_words.get(query_type, []):
                if w in tl:
                    score += 10
                if w in ab:
                    score += 3

            year = self._extract_year(a.get("pubdate", ""))
            if year:
                if year >= 2023:
                    score += 8
                elif year >= 2020:
                    score += 5
                elif year >= 2015:
                    score += 2

            jl = a.get("journal", "").lower()
            if any(j in jl for j in [
                "lancet", "bmj", "jama", "new england", "cochrane",
                "pediatrics", "annals", "nature", "plos"
            ]):
                score += 5

            a["relevance_score"] = score

        articles.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        return articles

    # ================================================================
    # PUBMED SEARCH API
    # ================================================================

    def _run_search(self, query, max_results):
        max_results = self._safe_int(max_results, 10, 1, 200)
        try:
            resp = requests.get(
                self.base_url + "/esearch.fcgi",
                params=self._api_params({
                    "db": "pubmed", "term": str(query),
                    "retmode": "json", "retmax": str(max_results),
                    "sort": "relevance",
                }),
                timeout=20,
            )
            resp.raise_for_status()
            es = resp.json().get("esearchresult", {})
            if "ERROR" in es:
                return []

            ids = es.get("idlist", es.get("IdList", []))
            if not ids:
                return []

            resp = requests.get(
                self.base_url + "/esummary.fcgi",
                params=self._api_params({
                    "db": "pubmed", "id": ",".join(ids), "retmode": "json"
                }),
                timeout=20,
            )
            resp.raise_for_status()
            sums = resp.json().get("result", {})

            articles = []
            for aid in ids:
                if aid not in sums or not isinstance(sums[aid], dict):
                    continue
                art = sums[aid]
                if "title" not in art:
                    continue
                articles.append({
                    "title": art.get("title", "Untitled"),
                    "authors": ", ".join(
                        a["name"] for a in art.get("authors", [])
                        if isinstance(a, dict) and a.get("name")
                    ),
                    "pubdate": art.get("pubdate", ""),
                    "journal": art.get("fulljournalname", ""),
                    "volume": art.get("volume", ""),
                    "issue": art.get("issue", ""),
                    "pages": art.get("pages", ""),
                    "doi": next(
                        (x["value"] for x in art.get("articleids", [])
                         if isinstance(x, dict) and x.get("idtype") == "doi"),
                        ""
                    ),
                    "pmid": aid,
                    "url": "https://pubmed.ncbi.nlm.nih.gov/" + aid + "/",
                    "abstract": "",
                    "ref_number": 0,
                })
            return articles
        except Exception:
            return []

    # ================================================================
    # VANCOUVER STYLE REFERENCE BUILDER
    # ================================================================

    def _vancouver_ref(self, article: Dict) -> str:
        """
        Format a single article as Vancouver style reference.
        Format: [N] Authors. Title. Journal. Year;Vol(Issue):Pages. doi:XX. PMID:XX.
        """

        ref_num = article.get("ref_number", 0)
        parts = []

        # Authors (Vancouver: up to 6, then et al.)
        authors = article.get("authors", "")
        if authors:
            auth_list = [a.strip() for a in authors.split(",") if a.strip()]
            if len(auth_list) > 6:
                auth_str = ", ".join(auth_list[:6]) + ", et al"
            else:
                auth_str = ", ".join(auth_list)
            parts.append(auth_str + ".")
        else:
            parts.append("[No authors listed].")

        # Title
        title = article.get("title", "Untitled").rstrip(".")
        parts.append(title + ".")

        # Journal
        journal = article.get("journal", "")
        if journal:
            parts.append(journal + ".")

        # Year;Volume(Issue):Pages
        year = self._extract_year(article.get("pubdate", ""))
        pub_detail = ""
        if year:
            pub_detail = str(year)
        vol = article.get("volume", "")
        if vol:
            if pub_detail:
                pub_detail += ";"
            pub_detail += vol
            issue = article.get("issue", "")
            if issue:
                pub_detail += "(" + issue + ")"
            pages = article.get("pages", "")
            if pages:
                pub_detail += ":" + pages
        if pub_detail:
            parts.append(pub_detail + ".")

        # DOI
        doi = article.get("doi", "")
        if doi:
            parts.append("doi:" + doi + ".")

        # PMID
        pmid = article.get("pmid", "")
        if pmid:
            parts.append("PMID: " + pmid + ".")

        ref_text = " ".join(parts)

        return "[" + str(ref_num) + "] " + ref_text

    def _build_vancouver_list(self, articles: List[Dict]) -> str:
        """Build a complete Vancouver-style numbered reference list"""
        md = ""
        for a in articles:
            md += self._vancouver_ref(a) + "\n\n"
        return md

    # ================================================================
    # FORMATTING: MAIN RESULTS
    # ================================================================

    def _format_results(self, query, analysis, query_type, search_log, articles, total, show_abs):
        md = "# 📚 PubMed Search Results\n\n"
        md += "**Question:** " + query + "\n\n"

        # Query understanding
        md += "## 🧠 Query Understanding\n\n"
        md += "**Type:** " + query_type.replace("_", " ").title() + "\n"
        if analysis["mesh_found"]:
            md += "**MeSH:** " + ", ".join(analysis["mesh_found"]) + "\n"
        if analysis["query_translation"]:
            md += "\n```\n" + analysis["query_translation"] + "\n```\n"
        md += "\n"

        # Search process
        md += "## 🔧 Search ("
        md += str(len(search_log)) + " strategies, "
        md += str(total) + " candidates)\n\n"
        for s in search_log:
            icon = "✅" if s["found"] > 0 else "⭕"
            md += icon + " **" + s["name"] + "** → " + str(s["found"]) + "  \n"
        md += "\n"

        # Results with abstracts
        with_abs = sum(1 for a in articles if a.get("abstract"))
        md += "## 📄 Top " + str(len(articles)) + " Results"
        if with_abs:
            md += " (" + str(with_abs) + " with abstracts)"
        md += "\n\n"

        md += self._format_article_list(articles, show_abs)

        # Vancouver reference list
        md += "## 📝 References (Vancouver Style)\n\n"
        md += self._build_vancouver_list(articles)

        # AI synthesis hint
        if with_abs:
            md += "## 🤖 AI Analysis Ready\n\n"
            md += "Abstracts are loaded. You can now ask:\n\n"
            md += "> Summarize the key findings from these articles\n\n"
            md += "> What is the current evidence on " + query + "?\n\n"
            md += "> Compare the conclusions across these studies\n\n"
            md += "When I cite findings, I will use the reference numbers above "
            md += "(e.g., [1], [2], [3]).\n\n"

        md += self._format_next_steps()
        return md

    def _format_article_list(self, articles, show_abstracts=True):
        md = ""
        for a in articles:
            ref = a.get("ref_number", 0)
            score = a.get("relevance_score", 0)
            stars = min(5, max(1, score // 5))

            md += "### [" + str(ref) + "] " + a.get("title", "Untitled") + "\n\n"

            if a.get("authors"):
                auth_list = a["authors"].split(", ")
                if len(auth_list) > 3:
                    auth_str = ", ".join(auth_list[:3]) + ", et al."
                else:
                    auth_str = a["authors"]
                md += "**Authors:** " + auth_str + "\n\n"

            info = []
            if a.get("journal"):
                info.append("*" + a["journal"] + "*")
            if a.get("pubdate"):
                info.append(a["pubdate"])
            v = a.get("volume", "")
            if v:
                if a.get("issue"):
                    v += "(" + a["issue"] + ")"
                if a.get("pages"):
                    v += ":" + a["pages"]
                info.append(v)
            if info:
                md += " | ".join(info) + "\n\n"

            links = ""
            if a.get("doi"):
                links += "[DOI](https://doi.org/" + a["doi"] + ") · "
            links += "[PMID " + a["pmid"] + "](" + a["url"] + ")"
            links += " · " + "⭐" * stars
            md += links + "\n\n"

            if show_abstracts and a.get("abstract"):
                md += "<details>\n<summary>📋 Abstract [" + str(ref) + "]</summary>\n\n"
                md += a["abstract"] + "\n\n</details>\n\n"

            md += "---\n\n"

        return md

    def _format_next_steps(self):
        return (
            "\n## 💡 Next Steps\n\n"
            "| Say | Get |\n|-----|-----|\n"
            "| `get results as list` | Vancouver reference list |\n"
            "| `get results as ris` | RIS file for Zotero |\n"
            "| `get results as summary` | AI synthesis of findings |\n"
            "| `get results as abstracts` | All abstracts for reading |\n"
            "| `get results as detailed` | Full metadata |\n\n"
        )

    def _format_no_results(self, query, analysis, search_log):
        md = "# No Results\n\n**Query:** " + query + "\n\n"
        if analysis["query_translation"]:
            md += "```\n" + analysis["query_translation"] + "\n```\n\n"
        for s in search_log:
            md += "❌ " + s["name"] + ": `" + s["query"] + "`\n\n"
        md += "Try simpler terms or `find_mesh`.\n"
        return md

    # ================================================================
    # OUTPUT FORMATS
    # ================================================================

    def _format_vancouver_list(self):
        """Numbered Vancouver reference list"""
        md = "# 📋 References (" + str(len(self._last_results)) + ")\n\n"
        md += "**Search:** " + self._last_query + "\n\n"
        md += self._build_vancouver_list(self._last_results)
        md += "\n> Say `get results as ris` for Zotero export\n"
        return md

    def _export_ris(self):
        ris = ""
        for a in self._last_results:
            ris += self._to_ris(a)
        return (
            "# 📥 RIS Export (" + str(len(self._last_results)) + " refs)\n\n"
            "1. Copy the code block\n"
            "2. Save as `references.ris`\n"
            "3. Zotero → File → Import\n\n"
            "```ris\n" + ris + "```\n"
        )

    def _format_abstracts_only(self):
        md = "# 📋 Abstracts (" + str(len(self._last_results)) + ")\n\n"
        md += "**Search:** " + self._last_query + "\n\n---\n\n"
        for a in self._last_results:
            ref = a.get("ref_number", 0)
            yr = self._extract_year(a.get("pubdate", "")) or "n.d."
            auth_list = a.get("authors", "").split(", ")
            first = auth_list[0] if auth_list and auth_list[0] else "Unknown"

            md += "## [" + str(ref) + "] " + a.get("title", "") + "\n"
            md += "*" + first + " et al. (" + str(yr) + ") — " + a.get("journal", "") + "*\n\n"

            if a.get("abstract"):
                md += a["abstract"] + "\n\n"
            else:
                md += "*No abstract available.*\n\n"

            md += "---\n\n"
        return md

    def _synthesize(self):
        articles = self._last_results
        md = "# 📊 Research Summary\n\n"
        md += "**Question:** " + self._last_query + "\n"
        md += "**Articles:** " + str(len(articles)) + "\n\n"

        years = [self._extract_year(a.get("pubdate", "")) for a in articles]
        years = [y for y in years if y]
        if years:
            md += "**Range:** " + str(min(years)) + "–" + str(max(years)) + "\n\n"

        with_abs = sum(1 for a in articles if a.get("abstract"))
        md += "**Abstracts available:** " + str(with_abs) + "/" + str(len(articles)) + "\n\n"

        # Journals
        journals = {}
        for a in articles:
            j = a.get("journal", "Unknown")
            journals[j] = journals.get(j, 0) + 1
        md += "## Sources\n\n"
        for j, c in sorted(journals.items(), key=lambda x: -x[1])[:8]:
            md += "- " + j + " (" + str(c) + ")\n"
        md += "\n"

        # Themes
        all_text = " ".join(
            a.get("title", "") + " " + a.get("abstract", "")
            for a in articles
        )
        wf = {}
        stops = {
            "the", "and", "for", "with", "from", "that", "this", "was", "were",
            "been", "have", "has", "study", "review", "patients", "results",
            "methods", "conclusion", "background", "objective", "clinical",
            "using", "based", "among", "between", "group", "data", "included",
            "also", "more", "than", "which", "were", "these", "other"
        }
        for w in re.findall(r"[a-z]{4,}", all_text.lower()):
            if w not in stops:
                wf[w] = wf.get(w, 0) + 1

        md += "## Key Themes\n\n"
        for w, c in sorted(wf.items(), key=lambda x: -x[1])[:15]:
            if c >= 3:
                md += "- **" + w + "** (" + str(c) + "×)\n"
        md += "\n"

        # Article summaries with reference numbers
        md += "## Articles\n\n"
        for a in articles[:20]:
            ref = a.get("ref_number", 0)
            yr = self._extract_year(a.get("pubdate", "")) or "n.d."
            auth_list = a.get("authors", "").split(", ")
            first = auth_list[0] if auth_list and auth_list[0] else "Unknown"

            md += "**[" + str(ref) + "]** " + first + " (" + str(yr) + "). "
            md += a.get("title", "") + " *" + a.get("journal", "") + "*\n"

            if a.get("abstract"):
                snippet = a["abstract"][:200]
                if len(a["abstract"]) > 200:
                    snippet += "..."
                md += "  " + snippet + "\n"
            md += "\n"

        md += "---\n"
        md += "*Cite using reference numbers: [1], [2], etc.*\n"
        return md

    def _format_detailed(self):
        md = "# 📑 Detailed (" + str(len(self._last_results)) + ")\n\n"
        for a in self._last_results:
            ref = a.get("ref_number", 0)
            md += "## [" + str(ref) + "] " + a.get("title", "") + "\n\n"
            md += "- **Authors:** " + a.get("authors", "Unknown") + "\n"
            md += "- **Journal:** " + a.get("journal", "Unknown") + "\n"
            md += "- **Date:** " + a.get("pubdate", "Unknown") + "\n"
            if a.get("doi"):
                md += "- **DOI:** [" + a["doi"] + "](https://doi.org/" + a["doi"] + ")\n"
            md += "- **PMID:** [" + a["pmid"] + "](" + a["url"] + ")\n"
            md += "- **Relevance:** " + str(a.get("relevance_score", 0))
            md += " · via " + a.get("found_via", "?") + "\n"
            if a.get("abstract"):
                md += "\n**Abstract:**\n\n" + a["abstract"] + "\n"
            md += "\n---\n\n"
        return md

    # ================================================================
    # UTILITIES
    # ================================================================

    def _to_ris(self, a):
        ris = "TY  - JOUR\n"
        if a.get("authors"):
            for au in a["authors"].split(", "):
                au = au.strip()
                if au:
                    ris += "AU  - " + au + "\n"
        title = a.get("title", "").rstrip(".")
        ris += "T1  - " + title + "\n"
        if a.get("journal"):
            ris += "JO  - " + a["journal"] + "\n"
        if a.get("pubdate"):
            m = re.search(r"(\d{4})", a["pubdate"])
            if m:
                ris += "PY  - " + m.group(1) + "\n"
            ris += "DA  - " + a["pubdate"] + "\n"
        if a.get("volume"):
            ris += "VL  - " + a["volume"] + "\n"
        if a.get("issue"):
            ris += "IS  - " + a["issue"] + "\n"
        if a.get("pages"):
            if "-" in a["pages"]:
                sp, ep = a["pages"].split("-", 1)
                ris += "SP  - " + sp.strip() + "\n"
                ris += "EP  - " + ep.strip() + "\n"
            else:
                ris += "SP  - " + a["pages"] + "\n"
        if a.get("doi"):
            ris += "DO  - " + a["doi"] + "\n"
        if a.get("url"):
            ris += "UR  - " + a["url"] + "\n"
        if a.get("abstract"):
            # Truncate very long abstracts for RIS
            abstract = a["abstract"][:2000]
            # Remove markdown formatting from abstract
            abstract = re.sub(r"\*\*[^*]+:\*\*\s*", "", abstract)
            ris += "AB  - " + abstract + "\n"
        ris += "ER  -\n\n"
        return ris

    def _extract_year(self, d):
        if not d:
            return None
        m = re.search(r"(\d{4})", str(d))
        return int(m.group(1)) if m else None

    def _safe_int(self, v, default=10, mn=1, mx=200):
        try:
            r = int(float(str(v)))
        except (TypeError, ValueError):
            r = default
        return max(mn, min(mx, r))

    def _error_msg(self, msg):
        return (
            "**Search Error:** " + msg + "\n\n"
            "Try:\n"
            "- Simpler phrasing\n"
            "- `find_mesh` to check terms\n"
            "- `pico_search` for structured queries\n"
        )
