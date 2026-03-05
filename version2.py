"""
title: Smart PubMed Research Assistant
author: Research Assistant
version: 6.2.0
date: 2025-01-01
license: MIT
description: Intelligent PubMed research assistant. Uses PubMed Automatic Term Mapping. Fetches abstracts with embedded citation numbers. Vancouver-style references. RIS export. Works without API key.
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
            description="Optional: NCBI API key from https://www.ncbi.nlm.nih.gov/account/settings/ — leave empty to work without it."
        )

    def __init__(self):
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.valves = self.Valves()
        self._cache = {}
        self._last_results = []
        self._last_query = ""

    def _api_params(self, params: dict) -> dict:
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
            description="Research question in plain English",
        ),
        max_results: int = Field(
            10,
            description="How many articles (1-200)",
        ),
        include_abstracts: bool = Field(
            True,
            description="Include abstracts for AI analysis",
        ),
    ) -> str:
        """
        Smart PubMed search. Returns numbered references with abstracts.
        After searching, ask the AI to summarize or analyze the findings.
        """
        try:
            max_results = self._safe_int(max_results, 10, 1, 200)
            query = str(query).strip()
            if not query:
                return "Please ask a research question."

            if isinstance(include_abstracts, str):
                include_abstracts = include_abstracts.lower() not in ("false", "no", "0")
            elif not isinstance(include_abstracts, bool):
                include_abstracts = True

            analysis = self._analyze_via_pubmed(query)
            query_type = self._detect_query_type(query)
            all_articles, search_log = self._iterative_search(
                query, analysis, query_type, max_results
            )

            if include_abstracts and all_articles:
                pmids = [a["pmid"] for a in all_articles]
                abstracts = self._fetch_abstracts(pmids)
                for article in all_articles:
                    article["abstract"] = abstracts.get(article["pmid"], "")

            scored = self._score_relevance(all_articles, query, query_type)
            top = scored[:max_results]

            for i, article in enumerate(top):
                article["ref_number"] = i + 1

            self._last_results = top
            self._last_query = query

            if not top:
                return self._format_no_results(query, analysis, search_log)

            return self._format_results(
                query, analysis, query_type, search_log,
                top, len(all_articles), include_abstracts
            )

        except Exception as e:
            return self._error_msg(str(e))

    # ================================================================
    # GET RESULTS
    # ================================================================

    def get_results(
        self,
        format: str = Field(
            "list",
            description="'list', 'ris', 'summary', 'abstracts', 'detailed'",
        ),
    ) -> str:
        """Get results in different formats."""
        try:
            fmt = str(format).strip().lower() if format else "list"
            if not self._last_results:
                return "No stored results. Run search_pubmed first."

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
        population: str = Field(..., description="Who?"),
        intervention: str = Field("", description="What treatment?"),
        comparison: str = Field("", description="Versus?"),
        outcome: str = Field("", description="What outcome?"),
        max_results: int = Field(15, description="How many (1-200)"),
    ) -> str:
        """PICO search with abstracts."""
        try:
            max_results = self._safe_int(max_results, 15, 1, 200)

            pico = {}
            for label, val in [("Population", population), ("Intervention", intervention),
                               ("Comparison", comparison), ("Outcome", outcome)]:
                val = str(val).strip() if val else ""
                if val:
                    pico[label] = val

            if not pico:
                return "Provide at least a Population."

            pico_analysis = {}
            for comp, text in pico.items():
                pico_analysis[comp] = self._analyze_via_pubmed(text)

            all_articles, search_log = self._pico_iterative_search(
                pico, pico_analysis, max_results
            )

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

            for i, a in enumerate(top):
                a["ref_number"] = i + 1

            self._last_results = top
            self._last_query = "PICO: " + "; ".join(k + "=" + v for k, v in pico.items())

            md = "# PICO Search Results\n\n"
            md += "| Component | Input | MeSH |\n|---|---|---|\n"
            for comp, text in pico.items():
                mapped = ", ".join(pico_analysis[comp].get("mesh_found", [])[:3]) or text
                md += "| " + comp + " | " + text + " | " + mapped + " |\n"
            md += "\n"

            if top:
                md += self._build_evidence_summary(top, combined)
            else:
                md += "No results found.\n"

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
        """Find MeSH terms."""
        try:
            analysis = self._analyze_via_pubmed(str(topic).strip())
            md = "# MeSH: " + topic + "\n\n"
            if analysis["mesh_found"]:
                md += "| Term | Syntax |\n|---|---|\n"
                for t in analysis["mesh_found"]:
                    md += "| " + t + " | `\"" + t + "\"[MeSH]` |\n"
                md += "\n"
            if analysis["query_translation"]:
                md += "```\n" + analysis["query_translation"] + "\n```\n"
            return md
        except Exception as e:
            return self._error_msg(str(e))

    # ================================================================
    # PUBMED QUERY ANALYSIS
    # ================================================================

    def _analyze_via_pubmed(self, query: str) -> Dict:
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
    # QUERY TYPE
    # ================================================================

    def _detect_query_type(self, query):
        q = query.lower()
        for qt, words in {
            "guidelines": ["guideline", "guidelines", "protocol", "recommendation", "consensus", "management of"],
            "systematic_review": ["systematic review", "meta-analysis"],
            "outcomes": ["outcome", "outcomes", "effectiveness", "efficacy"],
            "epidemiology": ["prevalence", "incidence", "epidemiology"],
            "diagnosis": ["diagnosis", "diagnostic", "screening"],
            "treatment": ["treatment", "therapy", "drug", "medication"],
            "risk_factors": ["risk factor", "cause", "etiology"],
        }.items():
            if any(w in q for w in words):
                return qt
        return "general"

    def _get_type_filter(self, qt):
        return {
            "guidelines": " AND (\"Practice Guideline\"[PT] OR \"Guideline\"[PT] OR guideline[ti])",
            "systematic_review": " AND (\"Systematic Review\"[PT] OR \"Meta-Analysis\"[PT])",
            "outcomes": " AND (\"Clinical Trial\"[PT] OR \"Comparative Study\"[PT])",
            "diagnosis": " AND (diagnosis[ti] OR diagnostic[ti])",
            "epidemiology": " AND (prevalence[ti] OR incidence[ti] OR epidemiology[sh])",
            "treatment": " AND (\"Clinical Trial\"[PT] OR \"Randomized Controlled Trial\"[PT])",
        }.get(qt, "")

    # ================================================================
    # STRATEGIES
    # ================================================================

    def _build_strategies(self, query, analysis, query_type):
        strategies = []
        mesh = analysis.get("mesh_found", [])
        count = analysis.get("result_count", 0)
        tf = self._get_type_filter(query_type)

        if count > 0:
            strategies.append(("PubMed Auto-Mapping", query))
        if mesh and tf:
            mt = ['"' + t + '"[MeSH]' for t in mesh[:4]]
            strategies.append(("MeSH + " + query_type, " AND ".join(mt) + tf))
        if mesh:
            mt = ['"' + t + '"[MeSH]' for t in mesh[:4]]
            strategies.append(("MeSH Combined", " AND ".join(mt)))
        if len(mesh) >= 2:
            strategies.append(("Core MeSH", '"' + mesh[0] + '"[MeSH] AND "' + mesh[1] + '"[MeSH]'))
        if mesh and tf:
            strategies.append(("Primary MeSH + " + query_type, '"' + mesh[0] + '"[MeSH]' + tf))
        strategies.append(("Title/Abstract", "(" + query + ")[tiab]"))
        strategies.append(("All Fields", query))
        return strategies

    def _iterative_search(self, query, analysis, query_type, max_results):
        return self._run_strategies(
            self._build_strategies(query, analysis, query_type), max_results
        )

    def _pico_iterative_search(self, pico, pico_analysis, max_results):
        strategies = []
        comp_parts = []
        for comp, analysis in pico_analysis.items():
            mesh = analysis.get("mesh_found", [])
            if mesh:
                if len(mesh) > 1:
                    mt = ['"' + t + '"[MeSH]' for t in mesh[:3]]
                    comp_parts.append("(" + " OR ".join(mt) + ")")
                else:
                    comp_parts.append('"' + mesh[0] + '"[MeSH]')
            else:
                comp_parts.append("(" + pico[comp] + ")")

        if len(comp_parts) >= 2:
            strategies.append(("Full PICO", " AND ".join(comp_parts)))
        combined = " ".join(pico.values())
        strategies.append(("Natural Language", combined))
        for pn, keys in [("P+I", ["Population", "Intervention"]), ("P+O", ["Population", "Outcome"])]:
            parts = []
            for k in keys:
                if k in pico_analysis:
                    mesh = pico_analysis[k].get("mesh_found", [])
                    if mesh:
                        parts.append('"' + mesh[0] + '"[MeSH]')
                    elif k in pico:
                        parts.append(pico[k])
            if len(parts) == 2:
                strategies.append((pn, " AND ".join(parts)))
        strategies.append(("Broad", combined))
        return self._run_strategies(strategies, max_results)

    def _run_strategies(self, strategies, max_results):
        all_articles = []
        seen = set()
        log = []
        for name, sq in strategies:
            if not sq:
                continue
            results = self._run_search(sq, min(max_results * 2, 200))
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

    def _fetch_abstracts(self, pmids):
        abstracts = {}
        if not pmids:
            return abstracts

        for i in range(0, len(pmids), 25):
            batch = pmids[i:i + 25]
            try:
                resp = requests.get(
                    self.base_url + "/efetch.fcgi",
                    params=self._api_params({
                        "db": "pubmed", "id": ",".join(batch),
                        "rettype": "xml", "retmode": "xml",
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

                for art in root.findall(".//PubmedArticle"):
                    pe = art.find(".//PMID")
                    if pe is None:
                        continue
                    pmid = pe.text
                    parts = []
                    ae = art.find(".//Abstract")
                    if ae is not None:
                        for txt in ae.findall("AbstractText"):
                            label = txt.get("Label", "")
                            text = self._elem_text(txt)
                            if text:
                                if label:
                                    parts.append(label + ": " + text)
                                else:
                                    parts.append(text)
                    if parts:
                        abstracts[pmid] = " ".join(parts)

                        mesh_list = [m.text for m in art.findall(".//MeshHeading/DescriptorName") if m.text]
                        if mesh_list:
                            abstracts[pmid] += " [MeSH: " + ", ".join(mesh_list[:8]) + "]"

            except Exception:
                continue
        return abstracts

    def _elem_text(self, elem):
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
        for pmid in pmids:
            if pmid in abstracts:
                continue
            pattern = r"<PMID[^>]*>" + re.escape(pmid) + r"</PMID>.*?<Abstract>(.*?)</Abstract>"
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
        qw = set(re.findall(r"[a-z]{3,}", query.lower())) - {
            "the", "and", "for", "with", "from", "that", "this", "are",
            "was", "were", "been", "have", "has", "how", "what", "which",
            "current", "recent", "new", "using", "based"
        }
        tw = {
            "guidelines": ["guideline", "guidelines", "recommendation", "consensus", "management"],
            "systematic_review": ["systematic", "review", "meta-analysis"],
            "outcomes": ["outcome", "outcomes", "effectiveness"],
            "diagnosis": ["diagnosis", "diagnostic", "screening"],
            "treatment": ["treatment", "therapy", "therapeutic"],
            "epidemiology": ["prevalence", "incidence", "epidemiology"],
        }
        for a in articles:
            s = 0
            tl = a.get("title", "").lower()
            ab = a.get("abstract", "").lower()
            s += len(qw & set(re.findall(r"[a-z]{3,}", tl))) * 5
            if ab:
                s += min(15, len(qw & set(re.findall(r"[a-z]{3,}", ab))) * 2)
                s += 3
            for w in tw.get(query_type, []):
                if w in tl: s += 10
                if w in ab: s += 3
            year = self._extract_year(a.get("pubdate", ""))
            if year:
                if year >= 2023: s += 8
                elif year >= 2020: s += 5
                elif year >= 2015: s += 2
            jl = a.get("journal", "").lower()
            if any(j in jl for j in ["lancet", "bmj", "jama", "new england", "cochrane", "pediatrics"]):
                s += 5
            a["relevance_score"] = s
        articles.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        return articles

    # ================================================================
    # SEARCH API
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
                params=self._api_params({"db": "pubmed", "id": ",".join(ids), "retmode": "json"}),
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
                         if isinstance(x, dict) and x.get("idtype") == "doi"), ""
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
    # KEY INNOVATION: EVIDENCE SUMMARY WITH EMBEDDED CITATIONS
    # Instead of separate instructions, we embed the citation
    # directly into the content the AI reads
    # ================================================================

    def _build_evidence_summary(self, articles, query):
        """
        Build a structured evidence block where each piece of
        information is ALREADY tagged with its reference number.
        The AI just reads this and naturally uses the numbers.
        """

        md = "## Evidence from " + str(len(articles)) + " articles\n\n"
        md += "Below is the evidence found. Each finding is tagged with its reference number.\n\n"

        for a in articles:
            ref = a.get("ref_number", 0)
            yr = self._extract_year(a.get("pubdate", "")) or "n.d."
            auth_list = a.get("authors", "").split(", ")
            first = auth_list[0] if auth_list and auth_list[0] else "Unknown"
            journal = a.get("journal", "")

            md += "---\n\n"
            md += "**REFERENCE [" + str(ref) + "]:** "
            md += first + " et al. (" + str(yr) + "). "
            md += a.get("title", "") + ". "
            md += "*" + journal + "*."
            if a.get("doi"):
                md += " doi:" + a["doi"]
            md += " PMID:" + a.get("pmid", "") + "\n\n"

            if a.get("abstract"):
                md += "**FINDINGS FROM [" + str(ref) + "]:** "
                md += a["abstract"] + "\n\n"
            else:
                md += "**[" + str(ref) + "]:** No abstract available.\n\n"

        md += "---\n\n"

        # Vancouver list
        md += "## Reference List\n\n"
        for a in articles:
            md += self._vancouver_ref(a) + "\n\n"

        # Short clear instruction
        md += "---\n\n"
        md += "IMPORTANT: When discussing any finding above, "
        md += "cite it as [1], [2], etc. matching the reference numbers. "
        md += "Example: 'Inhaled bronchodilators are first-line [1]. "
        md += "Oral corticosteroids reduce hospitalization [2,3].'\n\n"

        return md

    # ================================================================
    # MAIN FORMATTER
    # ================================================================

    def _format_results(self, query, analysis, query_type, search_log, articles, total, show_abs):
        md = "# PubMed Results: " + query + "\n\n"

        # Brief query info
        if analysis["mesh_found"]:
            md += "**MeSH:** " + ", ".join(analysis["mesh_found"]) + "\n"
        md += "**Found:** " + str(total) + " candidates → top " + str(len(articles)) + " shown\n\n"

        # The evidence summary with embedded citations
        md += self._build_evidence_summary(articles, query)

        md += self._format_next_steps()
        return md

    # ================================================================
    # VANCOUVER REFERENCE
    # ================================================================

    def _vancouver_ref(self, article):
        ref_num = article.get("ref_number", 0)
        parts = []

        authors = article.get("authors", "")
        if authors:
            al = [a.strip() for a in authors.split(",") if a.strip()]
            if len(al) > 6:
                parts.append(", ".join(al[:6]) + ", et al.")
            else:
                parts.append(", ".join(al) + ".")
        else:
            parts.append("[No authors].")

        parts.append(article.get("title", "Untitled").rstrip(".") + ".")
        if article.get("journal"):
            parts.append(article["journal"] + ".")

        yr = self._extract_year(article.get("pubdate", ""))
        pd = str(yr) if yr else ""
        vol = article.get("volume", "")
        if vol:
            if pd: pd += ";"
            pd += vol
            if article.get("issue"): pd += "(" + article["issue"] + ")"
            if article.get("pages"): pd += ":" + article["pages"]
        if pd:
            parts.append(pd + ".")

        if article.get("doi"):
            parts.append("doi:" + article["doi"] + ".")
        if article.get("pmid"):
            parts.append("PMID:" + article["pmid"] + ".")

        return "[" + str(ref_num) + "] " + " ".join(parts)

    def _build_vancouver_list(self, articles):
        md = ""
        for a in articles:
            md += self._vancouver_ref(a) + "\n\n"
        return md

    # ================================================================
    # NEXT STEPS
    # ================================================================

    def _format_next_steps(self):
        return (
            "\n## Next Steps\n\n"
            "| Command | Output |\n|---|---|\n"
            "| `get results as list` | Vancouver references |\n"
            "| `get results as ris` | Zotero RIS file |\n"
            "| `get results as summary` | Theme analysis |\n"
            "| `get results as abstracts` | All abstracts |\n"
            "| `get results as detailed` | Full metadata |\n\n"
        )

    def _format_no_results(self, query, analysis, search_log):
        md = "# No Results: " + query + "\n\n"
        if analysis["query_translation"]:
            md += "```\n" + analysis["query_translation"] + "\n```\n\n"
        for s in search_log:
            md += "- " + s["name"] + ": `" + s["query"] + "` (0 results)\n"
        md += "\nTry simpler terms.\n"
        return md

    # ================================================================
    # OUTPUT FORMATS
    # ================================================================

    def _format_vancouver_list(self):
        md = "# References (" + str(len(self._last_results)) + ")\n\n"
        md += self._build_vancouver_list(self._last_results)
        return md

    def _export_ris(self):
        ris = ""
        for a in self._last_results:
            ris += self._to_ris(a)
        return (
            "# RIS Export (" + str(len(self._last_results)) + ")\n\n"
            "Copy → save as .ris → Zotero Import\n\n"
            "```ris\n" + ris + "```\n"
        )

    def _format_abstracts_only(self):
        md = "# Abstracts (" + str(len(self._last_results)) + ")\n\n"
        for a in self._last_results:
            ref = a.get("ref_number", 0)
            md += "## [" + str(ref) + "] " + a.get("title", "") + "\n\n"
            if a.get("abstract"):
                md += a["abstract"] + "\n\n"
            else:
                md += "No abstract.\n\n"
            md += "---\n\n"
        return md

    def _synthesize(self):
        articles = self._last_results
        md = "# Summary: " + self._last_query + "\n\n"
        md += str(len(articles)) + " articles analyzed.\n\n"

        # Themes
        all_text = " ".join(a.get("title", "") + " " + a.get("abstract", "") for a in articles)
        wf = {}
        stops = {
            "the", "and", "for", "with", "from", "that", "this", "was", "were",
            "been", "have", "has", "study", "review", "patients", "results",
            "methods", "conclusion", "background", "objective", "clinical",
            "using", "based", "among", "between", "group", "data", "also"
        }
        for w in re.findall(r"[a-z]{4,}", all_text.lower()):
            if w not in stops:
                wf[w] = wf.get(w, 0) + 1

        md += "## Themes\n\n"
        for w, c in sorted(wf.items(), key=lambda x: -x[1])[:12]:
            if c >= 3:
                md += "- " + w + " (" + str(c) + "x)\n"
        md += "\n"

        # Evidence with embedded citations
        md += self._build_evidence_summary(articles, self._last_query)
        return md

    def _format_detailed(self):
        md = "# Detailed (" + str(len(self._last_results)) + ")\n\n"
        for a in self._last_results:
            ref = a.get("ref_number", 0)
            md += "## [" + str(ref) + "] " + a.get("title", "") + "\n\n"
            md += "- Authors: " + a.get("authors", "?") + "\n"
            md += "- Journal: " + a.get("journal", "?") + "\n"
            md += "- Date: " + a.get("pubdate", "?") + "\n"
            if a.get("doi"):
                md += "- DOI: " + a["doi"] + "\n"
            md += "- PMID: " + a["pmid"] + "\n"
            md += "- Score: " + str(a.get("relevance_score", 0)) + "\n"
            if a.get("abstract"):
                md += "\n" + a["abstract"] + "\n"
            md += "\n---\n\n"
        return md

    # ================================================================
    # UTILITIES
    # ================================================================

    def _to_ris(self, a):
        ris = "TY  - JOUR\n"
        if a.get("authors"):
            for au in a["authors"].split(", "):
                if au.strip():
                    ris += "AU  - " + au.strip() + "\n"
        ris += "T1  - " + a.get("title", "").rstrip(".") + "\n"
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
            ab = re.sub(r"\[MeSH:.*?\]", "", a["abstract"][:2000])
            ris += "AB  - " + ab.strip() + "\n"
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
        return "Error: " + msg + "\n\nTry simpler terms or find_mesh."
