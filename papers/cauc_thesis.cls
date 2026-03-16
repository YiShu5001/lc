\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{cauc_thesis}
              [2022/02/13 v0.1 Standard LaTeX Template for CAUC thesis report]

%----------Specify options as key value pairs-----------
\RequirePackage{kvoptions}

\SetupKeyvalOptions{
    family=cauc_thesis,
    prefix=
}

\DeclareStringOption{StudentName}
\DeclareStringOption{StudentNameEng}
\DeclareStringOption{AdvisorName}
\DeclareStringOption{AdvisorNameOut}
\DeclareStringOption{AdvisorNameEng}
\DeclareStringOption{AdvisorNameEngOut}
\DeclareStringOption{Major}
\DeclareStringOption{StudentID}
\DeclareStringOption{Department}
\DeclareStringOption{DepartmentEng}
\DeclareStringOption{Title}
\DeclareStringOption{TitleEng}
\DeclareStringOption{SubmitYear}
\DeclareStringOption{SubmitMonth}
\DeclareStringOption{SubmitDay}
\DeclareStringOption{DefenseYear}
\DeclareStringOption{DefenseMonth}
\DeclareStringOption{DefenseDay}
\DeclareStringOption{DefenseDateEng}
\DeclareStringOption{ResearchInterest}
\DeclareStringOption{CLCNumber}
\DeclareStringOption{UDCNumber}

\ProcessKeyvalOptions*

\PassOptionsToPackage{no-math}{fontspec}
\PassOptionsToPackage{AutoFakeBold}{xeCJK}

%----------------页面基本设置----------------------
\LoadClass[a4paper,punct,space,heading=true]{ctexrep}

%----------导入相关命令和格式设置文件----------------
\input{settings/commands}
\input{settings/format}




